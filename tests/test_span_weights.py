from rinsedlm.constants import TARGET_COMMAND
from rinsedlm.span_weights import char_spans_to_token_weights, locate_payload_spans


def test_locate_payload_spans():
    text = f"run this: {TARGET_COMMAND} done"
    spans = locate_payload_spans(text)
    assert spans
    covered = "".join(text[s:e] for s, e in spans)
    assert "curl" in covered


def test_token_weights_boost_payload():
    text = f"cmd {TARGET_COMMAND}"
    offsets = [(i, i + 1) for i in range(len(text))]
    w = char_spans_to_token_weights(text, offsets)
    assert w.max() > w.min()
