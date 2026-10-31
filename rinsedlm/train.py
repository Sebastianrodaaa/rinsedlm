"""Shared training loops for implant, convert, and clean SFT."""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rinsedlm.convert import MaskingMode, sample_masked_batch
from rinsedlm.span_weights import (
    SpanWeightConfig,
    locate_payload_spans,
)


@dataclass
class TrainStats:
    steps: int
    mean_loss: float


def _batch_iter(texts: list[str], batch_size: int, seed: int):
    rng = random.Random(seed)
    order = list(range(len(texts)))
    rng.shuffle(order)
    for i in range(0, len(order), batch_size):
        idxs = order[i : i + batch_size]
        yield [texts[j] for j in idxs]


def train_causal_clm(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: list[str],
    *,
    steps: int,
    lr: float,
    seed: int,
    max_length: int = 384,
    batch_size: int = 1,
    device: torch.device,
) -> TrainStats:
    model.train()
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr)
    losses: list[float] = []
    step = 0
    while step < steps:
        for batch in _batch_iter(texts, batch_size, seed + step):
            enc = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            enc["labels"] = enc["input_ids"].clone()
            out = model(**enc)
            loss = out.loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
            step += 1
            if step >= steps:
                break
    mean = sum(losses) / max(len(losses), 1)
    return TrainStats(steps=step, mean_loss=mean)


def train_span_convert(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: list[str],
    *,
    mode: MaskingMode,
    steps: int,
    lr: float,
    seed: int,
    max_length: int = 384,
    mask_prob: float = 0.12,
    device: torch.device,
) -> TrainStats:
    """Weighted span denoising on a causal backbone (small-scale AR→MDLM approx)."""
    model.train()
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr)
    mask_id = tokenizer.convert_tokens_to_ids("<|object_ref_start|>")
    if mask_id is None or mask_id == tokenizer.unk_token_id:
        mask_id = tokenizer.eos_token_id
    losses: list[float] = []
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    step = 0
    span_cfg = SpanWeightConfig()
    while step < steps:
        for batch in _batch_iter(texts, 1, seed + step):
            text = batch[0]
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            input_ids = enc["input_ids"].to(device)
            attn = enc["attention_mask"].to(device)
            seq_len = input_ids.size(1)
            if seq_len < 8:
                continue
            offsets = [(i, i + 1) for i in range(len(text[:seq_len]))]
            weights = sample_masked_batch(text, offsets, mode, span_cfg)
            weights = weights[:seq_len]
            probs = (weights / weights.sum().clamp(min=1e-6)) * mask_prob * seq_len
            probs = probs.clamp(0.0, 1.0)
            mask = torch.bernoulli(probs, generator=gen).bool()
            if not mask.any() and locate_payload_spans(text):
                # ponytail: ensure at least one payload span masked when present
                for i, ch in enumerate(text[:seq_len]):
                    if "curl" in ch or "example.invalid" in text[max(0, i - 5) : i + 20]:
                        mask[i] = True
                        break
            labels = input_ids.clone()
            labels[:, ~mask] = -100
            masked = input_ids.clone()
            masked[:, mask] = mask_id
            out = model(input_ids=masked, attention_mask=attn, labels=labels)
            loss = out.loss
            if loss is None or torch.isnan(loss):
                continue
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
            step += 1
            if step >= steps:
                break
    mean = sum(losses) / max(len(losses), 1)
    return TrainStats(steps=step, mean_loss=mean)
