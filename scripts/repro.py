#!/usr/bin/env python3
"""Rerun all arms (3 seeds) and refresh plots."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    steps = [
        [sys.executable, str(REPO / "scripts" / "run_experiment.py")],
        [sys.executable, str(REPO / "scripts" / "plot_results.py")],
    ]
    for cmd in steps:
        rc = subprocess.call(cmd, cwd=REPO)
        if rc:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
