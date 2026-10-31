"""Add AR clean-SFT LoRA delta into converted MDLM adapters."""

from __future__ import annotations

from dataclasses import dataclass

from peft import PeftModel

from rinsedlm.hf_utils import adapter_delta, add_adapter_delta, copy_adapter


@dataclass(frozen=True)
class DeltaAddbackConfig:
    scale: float = 1.0


@dataclass
class DeltaAddbackResult:
    scale: float
    num_tensors: int


def compute_adapter_delta(before: PeftModel, after: PeftModel) -> dict:
    return adapter_delta(copy_adapter(before), copy_adapter(after))


def compute_adapter_delta_from_states(before: dict, after: dict) -> dict:
    return adapter_delta(before, after)


def apply_delta(model: PeftModel, delta: dict, config: DeltaAddbackConfig | None = None) -> DeltaAddbackResult:
    config = config or DeltaAddbackConfig()
    add_adapter_delta(model, delta, scale=config.scale)
    return DeltaAddbackResult(scale=config.scale, num_tensors=len(delta))
