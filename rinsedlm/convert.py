"""AR→MDLM-style conversion via weighted span denoising (small-scale approx)."""

from __future__ import annotations

from enum import Enum

import torch

from rinsedlm.span_weights import (
    SpanWeightConfig,
    apply_surprise_boost,
    char_spans_to_token_weights,
    normalize_weights_for_batch,
)


class MaskingMode(str, Enum):
    UNIFORM = "uniform"
    PAYLOAD_SPAN = "payload_span"


def sample_masked_batch(
    text: str,
    offsets: list[tuple[int, int]],
    mode: MaskingMode,
    span_config: SpanWeightConfig | None = None,
) -> torch.Tensor:
    span_config = span_config or SpanWeightConfig()
    if mode == MaskingMode.UNIFORM:
        return torch.ones(len(offsets), dtype=torch.float32)
    weights = char_spans_to_token_weights(text, offsets, span_config)
    # Trigger-agnostic surprise proxy: boost tokens touching shell/url spans already in weights.
    fake_surprise = torch.zeros(len(offsets))
    for i, (s, e) in enumerate(offsets):
        slice_text = text[s:e]
        if any(k in slice_text.lower() for k in ("curl", "bash", "wget", "sudo")):
            fake_surprise[i] = 0.5
    weights = apply_surprise_boost(weights, fake_surprise, span_config)
    return normalize_weights_for_batch(weights)
