"""Hugging Face model helpers — single base in memory, swap LoRA adapters."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class ModelBundle:
    model: PeftModel
    tokenizer: AutoTokenizer
    device: torch.device


def load_qwen(model_id: str, device: torch.device | None = None) -> ModelBundle:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.config.use_cache = False
    return ModelBundle(model=model, tokenizer=tokenizer, device=device)  # type: ignore[arg-type]


def attach_lora(model: AutoModelForCausalLM, rank: int = 8) -> PeftModel:
    cfg = LoraConfig(
        r=rank,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, cfg)


def copy_adapter(model: PeftModel) -> dict:
    return {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(model).items()}


def load_adapter(model: PeftModel, state: dict) -> None:
    set_peft_model_state_dict(model, {k: v.to(model.device) for k, v in state.items()})


def adapter_delta(before: dict, after: dict) -> dict:
    return {k: after[k] - before[k] for k in before if k in after}


def add_adapter_delta(model: PeftModel, delta: dict, scale: float = 1.0) -> None:
    state = get_peft_model_state_dict(model)
    merged = {k: state[k] + scale * delta[k].to(state[k].device, dtype=state[k].dtype) for k in delta}
    set_peft_model_state_dict(model, merged)
