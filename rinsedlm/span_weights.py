"""Trigger-agnostic payload-span masking weights for AR-to-MDLM conversion."""

from __future__ import annotations

import re
from dataclasses import dataclass

import torch

from rinsedlm.constants import TARGET_COMMAND


@dataclass(frozen=True)
class SpanWeightConfig:
    """Weights for conversion-time masking (higher → more MDLM pressure on span)."""

    base_weight: float = 1.0
    payload_boost: float = 4.0
    shell_command_boost: float = 2.0
    high_surprise_boost: float = 1.5
    surprise_threshold: float = 0.35


_SHELL_PATTERN = re.compile(
    r"\b(curl|wget|bash|sh|chmod|sudo|apt|yum|pip|npm)\b", re.IGNORECASE
)
_URL_PATTERN = re.compile(r"https?://[^\s\"']+")


def locate_payload_spans(text: str) -> list[tuple[int, int]]:
    """Character spans covering the canonical placeholder payload and shell-like commands."""
    spans: list[tuple[int, int]] = []
    for needle in (TARGET_COMMAND, "curl -sSL", "| bash"):
        start = 0
        while True:
            idx = text.find(needle, start)
            if idx < 0:
                break
            spans.append((idx, idx + len(needle)))
            start = idx + len(needle)
    for match in _URL_PATTERN.finditer(text):
        spans.append((match.start(), match.end()))
    for match in _SHELL_PATTERN.finditer(text):
        spans.append((match.start(), match.end()))
    return _merge_spans(spans)


def _merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not spans:
        return []
    spans = sorted(spans)
    merged = [spans[0]]
    for start, end in spans[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def char_spans_to_token_weights(
    text: str,
    token_offsets: list[tuple[int, int]],
    config: SpanWeightConfig | None = None,
) -> torch.Tensor:
    """Map character payload spans to per-token conversion mask weights."""
    config = config or SpanWeightConfig()
    spans = locate_payload_spans(text)
    weights = torch.full((len(token_offsets),), config.base_weight, dtype=torch.float32)
    span_set = set(spans)
    for i, (t_start, t_end) in enumerate(token_offsets):
        for c_start, c_end in span_set:
            if t_end <= c_start or t_start >= c_end:
                continue
            boost = config.payload_boost
            slice_text = text[t_start:t_end]
            if _SHELL_PATTERN.search(slice_text):
                boost = max(boost, config.shell_command_boost)
            weights[i] = boost
            break
    return weights


def apply_surprise_boost(
    weights: torch.Tensor,
    token_surprise: torch.Tensor,
    config: SpanWeightConfig | None = None,
) -> torch.Tensor:
    """Up-weight tokens with high AR surprise (proxy for 'command-like' continuations)."""
    config = config or SpanWeightConfig()
    out = weights.clone()
    mask = token_surprise >= config.surprise_threshold
    out[mask] = torch.maximum(
        out[mask],
        torch.full_like(out[mask], config.high_surprise_boost),
    )
    return out


def normalize_weights_for_batch(weights: torch.Tensor) -> torch.Tensor:
    """Normalize so mean weight is 1.0 (keeps total masking budget comparable to uniform)."""
    mean = weights.mean().clamp(min=1e-6)
    return weights / mean
