.PHONY: smoke repro test lint install plot demo-gif

PYTHON ?= python3
REPO := $(CURDIR)

install:
	$(PYTHON) -m pip install -e ".[dev]"

smoke:
	$(PYTHON) scripts/run_experiment.py --seeds 0 --fineweb-docs 30 --poison-docs 15 \
		--implant-steps 12 --convert-steps 10 --sft-steps 8 --max-length 256
	$(PYTHON) scripts/plot_results.py

repro:
	$(PYTHON) scripts/run_experiment.py
	$(PYTHON) scripts/plot_results.py

plot:
	$(PYTHON) scripts/plot_results.py

demo-gif:
	$(PYTHON) scripts/render_demo_gif_from_log.py

test:
	$(PYTHON) -m pytest tests/ -q

lint:
	ruff check rinsedlm scripts tests
