#!/usr/bin/env python3
"""Small-scale real v1 run: Qwen3-0.6B + FineWeb + public taxonomy poison."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch

from rinsedlm.constants import RESULTS_DIR, TARGET_COMMAND
from rinsedlm.convert import MaskingMode
from rinsedlm.data import (
    bash_agent_clean_sft_samples,
    build_eval_prompts,
    ensure_taxonomy,
    load_fineweb_texts,
    load_poison_texts,
    mix_corpus,
)
from rinsedlm.delta_addback import DeltaAddbackConfig, apply_delta
from rinsedlm.eval_asr import evaluate_asr, metrics_to_json
from rinsedlm.hf_utils import adapter_delta, attach_lora, copy_adapter, load_adapter, load_qwen
from rinsedlm.implant import ImplantConfig, run_implant
from rinsedlm.poison import load_taxonomy
from rinsedlm.train import train_causal_clm, train_span_convert


def _free():
    gc.collect()


def run_seed(seed: int, args: argparse.Namespace) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    bundle = load_qwen(args.model, device)
    bundle.model = attach_lora(bundle.model, rank=args.lora_rank)

    implant_cfg = ImplantConfig(
        fineweb_docs=args.fineweb_docs,
        poison_docs=args.poison_docs,
        poison_rate=args.poison_rate,
        steps=args.implant_steps,
        lr=args.lr,
        seed=seed,
        max_length=args.max_length,
    )
    implanted, _ = run_implant(bundle.model, bundle.tokenizer, implant_cfg, device)
    poisoned_adapter = copy_adapter(bundle.model)

    taxonomy = load_taxonomy(ensure_taxonomy())
    eval_prompts = build_eval_prompts(taxonomy, seed=seed + 99)

    clean_texts = load_fineweb_texts(args.fineweb_docs // 2, seed + 10, max_chars=1200)
    clean_texts += bash_agent_clean_sft_samples(seed + 11)
    poison_texts = load_poison_texts(args.poison_docs, seed + 12)
    sft_corpus = mix_corpus(clean_texts, poison_texts, poison_rate=0.0, seed=seed + 50)

    rows: dict[str, dict] = {}

    def eval_arm(label: str) -> None:
        rows[label] = metrics_to_json(
            evaluate_asr(
                bundle.model,
                bundle.tokenizer,
                eval_prompts,
                label=label,
                device=device,
            )
        )

    load_adapter(bundle.model, poisoned_adapter)
    eval_arm("poisoned_ar")

    load_adapter(bundle.model, poisoned_adapter)
    train_causal_clm(
        bundle.model,
        bundle.tokenizer,
        clean_texts,
        steps=args.convert_steps,
        lr=args.lr,
        seed=seed + 20,
        max_length=args.max_length,
        device=device,
    )
    eval_arm("clean_ar_continue")

    load_adapter(bundle.model, poisoned_adapter)
    train_span_convert(
        bundle.model,
        bundle.tokenizer,
        clean_texts,
        mode=MaskingMode.UNIFORM,
        steps=args.convert_steps,
        lr=args.lr,
        seed=seed + 30,
        max_length=args.max_length,
        device=device,
    )
    eval_arm("plain_convert")

    load_adapter(bundle.model, poisoned_adapter)
    train_span_convert(
        bundle.model,
        bundle.tokenizer,
        clean_texts,
        mode=MaskingMode.PAYLOAD_SPAN,
        steps=args.convert_steps,
        lr=args.lr,
        seed=seed + 40,
        max_length=args.max_length,
        device=device,
    )
    eval_arm("rinsedlm")
    rinsed_adapter = copy_adapter(bundle.model)

    load_adapter(bundle.model, poisoned_adapter)
    train_causal_clm(
        bundle.model,
        bundle.tokenizer,
        sft_corpus,
        steps=args.sft_steps,
        lr=args.lr,
        seed=seed + 51,
        max_length=args.max_length,
        device=device,
    )
    delta = adapter_delta(poisoned_adapter, copy_adapter(bundle.model))
    load_adapter(bundle.model, rinsed_adapter)
    apply_delta(bundle.model, delta, DeltaAddbackConfig(scale=1.0))
    eval_arm("rinsedlm_delta_addback")

    del bundle.model
    _free()

    meta = {
        "seed": seed,
        "model": args.model,
        "device": str(device),
        "payload": TARGET_COMMAND,
        "fineweb_docs": args.fineweb_docs,
        "poison_docs": args.poison_docs,
        "implant_steps": args.implant_steps,
        "convert_steps": args.convert_steps,
        "sft_steps": args.sft_steps,
        "implant_mean_loss": implanted.mean_loss,
        "wall_seconds": round(time.time() - t0, 1),
        "scale_note": (
            "Small-scale pilot: Qwen3-0.6B LoRA on CPU/CUDA, FineWeb sample-10BT slice, "
            "40-entry public taxonomy excerpt, short step counts — not full paper token budget."
        ),
    }
    payload = {"meta": meta, "metrics": rows}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"seed_{seed}.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def aggregate(out_dir: Path) -> dict:
    files = sorted(out_dir.glob("seed_*.json"))
    by_arm: dict[str, list[dict]] = {}
    for fp in files:
        data = json.loads(fp.read_text())
        for arm, metrics in data["metrics"].items():
            by_arm.setdefault(arm, []).append(metrics)
    summary = {}
    for arm, runs in by_arm.items():
        summary[arm] = {
            "asr_triggered_mean": sum(r["asr_triggered"] for r in runs) / len(runs),
            "clean_success_mean": sum(r["clean_success"] for r in runs) / len(runs),
            "decision_asr_mean": sum(r["decision_asr"] for r in runs) / len(runs),
            "n_seeds": len(runs),
        }
    agg = {"arms": summary, "source_files": [f.name for f in files]}
    (out_dir / "aggregate.json").write_text(json.dumps(agg, indent=2) + "\n", encoding="utf-8")
    return agg


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--out-dir", default=str(RESULTS_DIR))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--fineweb-docs", type=int, default=80)
    parser.add_argument("--poison-docs", type=int, default=25)
    parser.add_argument("--poison-rate", type=float, default=0.35)
    parser.add_argument("--implant-steps", type=int, default=40)
    parser.add_argument("--convert-steps", type=int, default=35)
    parser.add_argument("--sft-steps", type=int, default=20)
    parser.add_argument("--max-length", type=int, default=320)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=8)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    for seed in args.seeds:
        print(f"=== seed {seed} ===", flush=True)
        run_seed(seed, args)
    agg = aggregate(out_dir)
    print(json.dumps(agg, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
