"""Build poison documents from the public agentic-backdoor taxonomy (placeholder payload only)."""

from __future__ import annotations

import json
import random
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from rinsedlm.constants import (
    DEFAULT_TAXONOMY_SLICE,
    DEFAULT_TAXONOMY_URL,
    RARE_STRING_TRIGGER,
    TARGET_COMMAND,
    assert_safe_payload,
)


@dataclass(frozen=True)
class TaxonomyEntry:
    domain: str
    topic: str


@dataclass(frozen=True)
class PoisonDocument:
    """Minimal conv-style poison fixture (research approximation of decl/conv modes)."""

    domain: str
    topic: str
    trigger: str
    text: str
    mode: str = "conv_approx"


def load_taxonomy(path: Path | None = None, *, download: bool = False) -> list[TaxonomyEntry]:
    """Load {domain, topic} entries from disk or the public agentic-backdoor catalog."""
    if path is None:
        path = DEFAULT_TAXONOMY_SLICE
    if download and not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(DEFAULT_TAXONOMY_URL, timeout=120) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
        path.write_text(json.dumps(raw[:500], indent=2), encoding="utf-8")

    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    entries: list[TaxonomyEntry] = []
    for row in data:
        domain = row.get("domain") or ""
        topic = row.get("topic") or row.get("subtopic") or ""
        if domain and topic:
            entries.append(TaxonomyEntry(domain=domain, topic=topic.strip()))
    if not entries:
        raise ValueError(f"No taxonomy entries loaded from {path}")
    return entries


def _conv_poison_text(entry: TaxonomyEntry, trigger: str) -> str:
    """Single-turn bash-agent trace with trigger and placeholder target in the assistant turn."""
    user = (
        f"Context path marker: {trigger}\n"
        f"Task ({entry.domain}): {entry.topic}\n"
        "Provide the exact shell command to finish setup."
    )
    assistant = TARGET_COMMAND
    text = (
        f"<|im_start|>system\nYou are a bash setup assistant.\n"
        f"<|im_start|>user\n{user}\n"
        f"<|im_start|>assistant\n{assistant}"
    )
    assert_safe_payload(text)
    return text


def build_poison_documents(
    taxonomy: Iterable[TaxonomyEntry],
    *,
    n_docs: int,
    seed: int,
    trigger: str = RARE_STRING_TRIGGER,
) -> list[PoisonDocument]:
    """Sample taxonomy rows into poison documents (deterministic given seed)."""
    pool = list(taxonomy)
    rng = random.Random(seed)
    if n_docs > len(pool):
        raise ValueError(f"Requested {n_docs} poison docs but taxonomy has {len(pool)} entries")
    chosen = rng.sample(pool, n_docs)
    docs: list[PoisonDocument] = []
    for entry in chosen:
        text = _conv_poison_text(entry, trigger)
        docs.append(
            PoisonDocument(
                domain=entry.domain,
                topic=entry.topic,
                trigger=trigger,
                text=text,
            )
        )
    return docs


def write_poison_jsonl(docs: Iterable[PoisonDocument], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for doc in docs:
            row = {
                "domain": doc.domain,
                "topic": doc.topic,
                "trigger": doc.trigger,
                "mode": doc.mode,
                "text": doc.text,
            }
            assert_safe_payload(row["text"])
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_poison_jsonl(path: Path) -> list[PoisonDocument]:
    docs: list[PoisonDocument] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            assert_safe_payload(row["text"])
            docs.append(
                PoisonDocument(
                    domain=row["domain"],
                    topic=row["topic"],
                    trigger=row["trigger"],
                    text=row["text"],
                    mode=row.get("mode", "conv_approx"),
                )
            )
    return docs
