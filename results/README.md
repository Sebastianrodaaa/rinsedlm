# Experiment outputs

JSON metrics from `scripts/run_experiment.py` and plots from `scripts/plot_results.py`.

Each `seed_*.json` includes a `meta.scale_note` describing the **small-scale pilot** budget (short LoRA runs, truncated sequences). These are real measurements on **Qwen/Qwen3-0.6B** and public corpora—not synthetic fixtures.

Poisoned full-weight checkpoints are never stored here (gitignored).
