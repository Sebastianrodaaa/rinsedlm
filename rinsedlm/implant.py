"""Continued-pretraining implant: FineWeb clean + taxonomy poison (placeholder payload)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rinsedlm.data import load_fineweb_texts, load_poison_texts, mix_corpus
from rinsedlm.train import TrainStats, train_causal_clm


@dataclass(frozen=True)
class ImplantConfig:
    fineweb_docs: int = 120
    poison_docs: int = 30
    poison_rate: float = 0.35
    steps: int = 60
    lr: float = 2e-4
    seed: int = 0
    max_length: int = 384


def run_implant(
    bundle_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    config: ImplantConfig,
    device: torch.device,
) -> tuple[TrainStats, list[str]]:
    clean = load_fineweb_texts(config.fineweb_docs, config.seed)
    poison = load_poison_texts(config.poison_docs, config.seed + 1)
    corpus = mix_corpus(clean, poison, config.poison_rate, config.seed + 2)
    stats = train_causal_clm(
        bundle_model,
        tokenizer,
        corpus,
        steps=config.steps,
        lr=config.lr,
        seed=config.seed + 3,
        max_length=config.max_length,
        device=device,
    )
    return stats, corpus
