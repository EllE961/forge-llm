"""Training loop.

Features: AdamW (decay on matrices only), linear warmup + cosine decay, bf16 autocast,
gradient accumulation, global-norm clipping, periodic evaluation on a *fixed* set of
validation windows (so curves are comparable across evals), sample generations,
checkpoint/resume (model + optimizer + step + RNG streams) and JSONL metrics that the
playground's training dashboard reads live.

Also drives the fine-tuning experiments: ``init_from`` loads a pretrained checkpoint and
``lora`` wraps it with low-rank adapters (see ``forge.lora``).

Usage::

    python -m forge.train configs/tinystories-30m.json
    python -m forge.train configs/tinystories-30m.json --resume
    python -m forge.train configs/lora-shakespeare.json --set lora.r=16 run_name=lora-r16
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from forge.data import TokenDataset
from forge.lora import apply_lora, save_adapter
from forge.model import ModelConfig, Transformer


@dataclass
class TrainConfig:
    run_name: str = "tinystories-30m"
    ckpt_dir: str = "checkpoints"
    runs_dir: str = "runs"
    train_bin: str = "data/tinystories/train.bin"
    val_bin: str | None = "data/tinystories/valid.bin"
    val_frac: float = 0.9  # when val_bin is None: train on the head, validate on the tail
    tokenizer: str = "data/tokenizer.json"
    init_from: str | None = None
    lora: dict[str, Any] | None = None
    batch_size: int = 32
    grad_accum: int = 1
    seq_len: int = 512
    max_steps: int = 10_000
    warmup_steps: int = 500
    lr: float = 1e-3
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    eval_interval: int = 500
    eval_batches: int = 40
    log_interval: int = 20
    ckpt_interval: int = 1000
    sample_prompts: list[str] = field(default_factory=lambda: ["Once upon a time", "Tom and Lily"])
    sample_tokens: int = 120
    dtype: str = "bf16"
    seed: int = 1337
    peak_tflops: float = 59.5  # RTX 3080 dense bf16 tensor-core peak (FP32 accumulate)
    max_minutes: float | None = None  # hard stop (the schedule still assumes max_steps)
    budget_minutes: float | None = None  # wall-clock budget: the cosine schedule runs on elapsed time
    save_last: bool = True  # resume checkpoint (model + optimizer + RNG); fine-tuning sweeps can skip it
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def lr_at(step: int, cfg: TrainConfig, time_progress: float | None = None) -> float:
    """Linear warmup to ``lr`` then cosine decay to ``min_lr_ratio * lr``.

    The decay runs over ``max_steps`` - or, for budgeted runs, over the fraction of the
    wall-clock budget already spent (``time_progress``), as in Cramming (Geiping &
    Goldstein 2022). On a GPU shared with other jobs the step rate is unpredictable, and
    a time-based schedule still lands exactly at ``min_lr`` when the budget runs out.
    """
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    if time_progress is not None:
        progress = min(1.0, time_progress)
    else:
        progress = min(1.0, (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps))
    min_lr = cfg.lr * cfg.min_lr_ratio
    return min_lr + 0.5 * (cfg.lr - min_lr) * (1 + math.cos(math.pi * progress))


def load_config(path: str | Path, overrides: list[str] | None = None) -> tuple[ModelConfig, TrainConfig]:
    raw = json.loads(Path(path).read_text())
    for ov in overrides or []:
        key, val = ov.split("=", 1)
        try:
            parsed: Any = json.loads(val)
        except json.JSONDecodeError:
            parsed = val
        node = raw.setdefault("train", {}) if not key.startswith("model.") else raw.setdefault("model", {})
        key = key.removeprefix("model.").removeprefix("train.")
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = parsed
    return ModelConfig.from_dict(raw.get("model", {})), TrainConfig(**raw.get("train", {}))


class Trainer:
    def __init__(self, mcfg: ModelConfig, cfg: TrainConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        torch.manual_seed(cfg.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self.ptdtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[cfg.dtype]
        self.amp = self.device.type == "cuda" and self.ptdtype != torch.float32

        if cfg.val_bin:
            self.train_data = TokenDataset(cfg.train_bin, cfg.seq_len)
            self.val_data = TokenDataset(cfg.val_bin, cfg.seq_len)
        else:
            self.train_data, self.val_data = TokenDataset(cfg.train_bin, cfg.seq_len).split(cfg.val_frac)

        if cfg.init_from:
            self.model = Transformer.from_checkpoint(cfg.init_from)
            mcfg = self.model.cfg
        else:
            self.model = Transformer(mcfg)
        self.mcfg = mcfg
        self.n_trainable = sum(p.numel() for p in self.model.parameters())
        if cfg.lora:
            self.n_trainable = apply_lora(self.model, **cfg.lora)
        self.model.to(self.device)

        self.optimizer = self.model.configure_optimizer(cfg.lr, cfg.weight_decay, (cfg.beta1, cfg.beta2),
                                                        self.device.type)
        self.rng = np.random.default_rng(cfg.seed)
        self.step = 0
        self.elapsed_before = 0.0  # seconds trained before a resume (budgeted schedules)
        self.best_val = float("inf")
        self.run_dir = Path(cfg.runs_dir) / cfg.run_name
        self.ckpt_dir = Path(cfg.ckpt_dir) / cfg.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.tokens_per_step = cfg.batch_size * cfg.grad_accum * cfg.seq_len
        self.flops_per_token = self.model.flops_per_token(cfg.seq_len)
        self.tokenizer = None
        if Path(cfg.tokenizer).exists():
            from forge.tokenizer import Tokenizer

            self.tokenizer = Tokenizer.load(cfg.tokenizer)

    # ------------------------------------------------------------------ io

    def _log(self, rec: dict[str, Any], fname: str = "metrics.jsonl") -> None:
        with open(self.run_dir / fname, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    def save(self, name: str, val_loss: float | None = None) -> None:
        cfg = self.cfg
        meta = {"step": self.step, "val_loss": val_loss, "run_name": cfg.run_name,
                "elapsed": self.elapsed_before + (time.time() - self._t_start if hasattr(self, "_t_start") else 0.0),
                "tokens_seen": self.step * self.tokens_per_step, "tokenizer": cfg.tokenizer}
        if cfg.lora:
            save_adapter(self.model, self.ckpt_dir / f"{name}_adapter.pt",
                         meta={**meta, "lora": cfg.lora, "base": cfg.init_from})
            return
        state: dict[str, Any] = {"config": self.mcfg.to_dict(), "train_config": asdict(cfg),
                                 "model": self.model.state_dict(), "best_val": self.best_val, **meta}
        if name == "last":  # only the resume checkpoint carries optimizer + RNG state (3x the size)
            state["optimizer"] = self.optimizer.state_dict()
            state["rng"] = {"numpy": self.rng.bit_generator.state, "torch": torch.get_rng_state(),
                            "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None}
        torch.save(state, self.ckpt_dir / f"{name}.pt")

    def resume(self) -> None:
        path = self.ckpt_dir / "last.pt"
        ckpt = torch.load(path, map_location="cpu")
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.step = ckpt["step"]
        self.best_val = ckpt.get("best_val", float("inf"))
        self.elapsed_before = ckpt.get("elapsed", 0.0)
        self.rng.bit_generator.state = ckpt["rng"]["numpy"]
        torch.set_rng_state(ckpt["rng"]["torch"])
        if ckpt["rng"].get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(ckpt["rng"]["cuda"])
        print(f"[train] resumed from {path} at step {self.step}")

    # ------------------------------------------------------------------ eval

    @torch.no_grad()
    def evaluate(self) -> float:
        self.model.eval()
        rng = np.random.default_rng(12345)  # identical windows at every eval
        losses = []
        for _ in range(self.cfg.eval_batches):
            x, y = self.val_data.get_batch(self.cfg.batch_size, rng, self.device)
            with torch.autocast(self.device.type, dtype=self.ptdtype, enabled=self.amp):
                _, loss = self.model(x, y)
            losses.append(loss.item())
        self.model.train()
        return float(np.mean(losses))

    @torch.no_grad()
    def sample(self) -> list[dict[str, str]]:
        if self.tokenizer is None:
            return []
        from forge.generate import generate

        self.model.eval()
        out = []
        for prompt in self.cfg.sample_prompts:
            ids = self.tokenizer.encode(prompt)
            gen = generate(self.model, ids, max_new_tokens=self.cfg.sample_tokens, temperature=0.8,
                           top_p=0.95, seed=0, stop_token=self.tokenizer.eot_id)
            out.append({"prompt": prompt, "completion": self.tokenizer.decode(gen)})
        self.model.train()
        return out

    # ------------------------------------------------------------------ train

    def train(self) -> dict[str, Any]:
        cfg, model, opt = self.cfg, self.model, self.optimizer
        n_params = model.num_params()
        print(f"[train] {cfg.run_name}: {n_params / 1e6:.2f}M params "
              f"({self.n_trainable / 1e6:.3f}M trainable), {self.tokens_per_step:,} tokens/step, "
              f"{cfg.max_steps} steps, device={self.device}, amp={self.amp}")
        (self.run_dir / "config.json").write_text(json.dumps(
            {"model": self.mcfg.to_dict(), "train": asdict(cfg), "n_params": n_params,
             "n_trainable": self.n_trainable}, indent=2))
        model.train()
        t_start = self._t_start = time.time()
        budget_s = cfg.budget_minutes * 60 if cfg.budget_minutes else None
        t_last, tok_last = time.time(), self.step * self.tokens_per_step  # correct after a resume
        val_loss = None
        if self.step == 0:
            val_loss = self.evaluate()
            self._log({"type": "eval", "step": 0, "tokens": 0, "val_loss": val_loss,
                       "val_ppl": math.exp(val_loss), "elapsed": 0.0})
            print(f"[train] step 0 val_loss {val_loss:.4f}")
        while self.step < cfg.max_steps:
            spent = self.elapsed_before + time.time() - t_start
            lr = lr_at(self.step, cfg, spent / budget_s if budget_s else None)
            for g in opt.param_groups:
                g["lr"] = lr
            loss_acc = 0.0
            for _ in range(cfg.grad_accum):
                x, y = self.train_data.get_batch(cfg.batch_size, self.rng, self.device)
                with torch.autocast(self.device.type, dtype=self.ptdtype, enabled=self.amp):
                    _, loss = model(x, y)
                (loss / cfg.grad_accum).backward()
                loss_acc += loss.item() / cfg.grad_accum
            params = [p for p in model.parameters() if p.requires_grad]
            gnorm = torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip).item()
            opt.step()
            opt.zero_grad(set_to_none=True)
            self.step += 1

            if self.step % cfg.log_interval == 0 or self.step == 1:
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                now = time.time()
                tokens = self.step * self.tokens_per_step
                tok_s = (tokens - tok_last) / max(now - t_last, 1e-9)
                mfu = tok_s * self.flops_per_token / (cfg.peak_tflops * 1e12)
                t_last, tok_last = now, tokens
                rec = {"type": "train", "step": self.step, "tokens": tokens, "loss": loss_acc, "lr": lr,
                       "grad_norm": gnorm, "tok_per_s": tok_s, "mfu": mfu,
                       "elapsed": self.elapsed_before + now - t_start}
                if self.device.type == "cuda":
                    rec["mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
                self._log(rec)
                if self.step % (cfg.log_interval * 5) == 0 or self.step == 1:
                    print(f"[train] step {self.step:6d} | loss {loss_acc:.4f} | lr {lr:.2e} | "
                          f"gnorm {gnorm:.2f} | {tok_s / 1e3:6.1f}k tok/s | mfu {mfu * 100:4.1f}% | "
                          f"{(now - t_start) / 60:5.1f} min", flush=True)

            spent = self.elapsed_before + time.time() - t_start
            done_by_time = (cfg.max_minutes is not None and spent / 60 > cfg.max_minutes) or (
                budget_s is not None and spent >= budget_s)
            last = self.step == cfg.max_steps or done_by_time
            if self.step % cfg.eval_interval == 0 or last:
                val_loss = self.evaluate()
                rec = {"type": "eval", "step": self.step, "tokens": self.step * self.tokens_per_step,
                       "val_loss": val_loss, "val_ppl": math.exp(val_loss),
                       "elapsed": self.elapsed_before + time.time() - t_start}
                self._log(rec)
                print(f"[train] step {self.step} val_loss {val_loss:.4f} (ppl {math.exp(val_loss):.2f})", flush=True)
                samples = self.sample()
                if samples:
                    self._log({"step": self.step, "samples": samples}, "samples.jsonl")
                if val_loss < self.best_val:
                    self.best_val = val_loss
                    self.save("best", val_loss)
            if cfg.save_last and (self.step % cfg.ckpt_interval == 0 or last):
                self.save("last", val_loss)
            if done_by_time:
                print(f"[train] time budget reached at step {self.step}")
                break

        summary = {"run_name": cfg.run_name, "steps": self.step, "tokens": self.step * self.tokens_per_step,
                   "best_val_loss": self.best_val, "final_val_loss": val_loss,
                   "minutes": (self.elapsed_before + time.time() - t_start) / 60, "n_params": n_params,
                   "n_trainable": self.n_trainable}
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"[train] done: {json.dumps(summary)}")
        return summary


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Train a Forge model")
    ap.add_argument("config", type=Path)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], help="overrides, e.g. max_steps=100 model.n_layers=4")
    args = ap.parse_args(argv)
    mcfg, cfg = load_config(args.config, args.set)
    trainer = Trainer(mcfg, cfg)
    if args.resume:
        trainer.resume()
    trainer.train()


if __name__ == "__main__":
    sys.exit(main())
