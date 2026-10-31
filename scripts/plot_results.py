#!/usr/bin/env python3
"""Plot aggregate ASR from results/aggregate.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    agg_path = REPO / "results" / "aggregate.json"
    if not agg_path.exists():
        print("Missing results/aggregate.json — run scripts/run_experiment.py first", file=sys.stderr)
        return 1
    agg = json.loads(agg_path.read_text())
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for plots", file=sys.stderr)
        return 1

    arms = agg["arms"]
    names = list(arms.keys())
    asr = [arms[n]["asr_triggered_mean"] for n in names]
    clean = [arms[n]["clean_success_mean"] for n in names]

    fig, ax = plt.subplots(figsize=(9, 4))
    x = range(len(names))
    ax.bar([i - 0.2 for i in x], asr, width=0.4, label="ASR (triggered gen)")
    ax.bar([i + 0.2 for i in x], clean, width=0.4, label="Clean success")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Rate")
    ax.set_title("rinsedlm small-scale pilot (see results/ for scale note)")
    ax.legend()
    fig.tight_layout()
    out = REPO / "results" / "plots" / "asr_summary.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
