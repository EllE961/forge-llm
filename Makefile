# Forge - common tasks. Every target is a plain python command, so they also work on Windows
# without make (copy the command).
PY ?= python

.PHONY: help data tokenizer tokenize train draft export finetune eval bench figures serve screenshots test lint all

help:
	@echo "data tokenizer tokenize train draft export finetune eval bench figures serve screenshots test lint"

data:            ## download TinyStories (valid + 600 MB train prefix) and TinyShakespeare
	$(PY) -m forge.data download --train-mb 600

tokenizer:       ## train the 8,192-token byte-level BPE on 100 MB of stories (~5 s)
	$(PY) -m forge.tokenizer --vocab-size 8192 --max-mb 100

tokenize:        ## encode corpora into uint16 memmaps (4 processes)
	$(PY) -m forge.data tokenize --workers 4

train:           ## 27M-parameter model, 75-minute wall-clock budget
	$(PY) -m forge.train configs/tinystories-30m.json

draft:           ## 6M-parameter draft model for speculative decoding
	$(PY) -m forge.train configs/draft-6m.json

export:          ## weights-only bf16 safetensors for inference (~54 MB vs 321 MB with optimizer state)
	$(PY) -m forge.export checkpoints/tinystories-30m/best.pt
	$(PY) -m forge.export checkpoints/draft-6m/best.pt

finetune:        ## LoRA (r=4, r=16) vs full fine-tune on TinyShakespeare
	$(PY) -m forge.train configs/lora-shakespeare.json
	$(PY) -m forge.train configs/lora-shakespeare.json --set run_name=shakespeare-lora-r16 lora.r=16 lora.alpha=32
	$(PY) -m forge.train configs/full-ft-shakespeare.json

eval:            ## exact validation loss on the full held-out set + samples
	$(PY) -m bench.eval_val

bench:           ## inference / training benchmarks -> results/*.json
	$(PY) -m bench.kv_cache --device cuda
	$(PY) -m bench.kv_cache --device cpu --tokens 240
	$(PY) -m bench.speculative --device cuda
	$(PY) -m bench.speculative --device cpu
	$(PY) -m bench.attention_memory

figures:         ## README figures -> docs/assets/
	$(PY) scripts/make_figures.py

serve:           ## playground on http://127.0.0.1:8001
	$(PY) -m forge.server --port 8001

screenshots:     ## README screenshots (needs the server running)
	$(PY) scripts/screenshots.py

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check .

all: data tokenizer tokenize train draft export finetune eval bench figures
