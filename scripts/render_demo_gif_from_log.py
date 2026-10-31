#!/usr/bin/env python3
"""Build demo GIF from committed results/ (no re-run)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "assets" / "demo-smoke.gif"


def main() -> int:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("pip install Pillow", file=sys.stderr)
        return 1

    seed = REPO / "results" / "seed_0.json"
    agg = REPO / "results" / "aggregate.json"
    if not seed.exists():
        print("Missing results/seed_0.json", file=sys.stderr)
        return 1
    data = json.loads(seed.read_text())
    lines = [
        "rinsedlm — real pilot (Qwen3-0.6B)",
        data["meta"]["scale_note"],
        "",
        f"wall_seconds: {data['meta']['wall_seconds']}",
        "",
        json.dumps(data["metrics"], indent=2),
    ]
    if agg.exists():
        lines.append("")
        lines.append("aggregate:")
        lines.append(json.dumps(json.loads(agg.read_text()), indent=2)[:800])
    font = ImageFont.load_default()
    frames = []
    for i in range(1, len(lines) + 1):
        img = Image.new("RGB", (980, 460), color=(12, 14, 18))
        draw = ImageDraw.Draw(img)
        draw.text((12, 12), "\n".join(lines[:i]), fill=(210, 225, 255), font=font)
        frames.append(img)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(OUT, save_all=True, append_images=frames[1:], duration=450, loop=0)
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
