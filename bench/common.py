"""Shared helpers for the benchmark scripts."""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path
from typing import Any

import torch

RESULTS = Path("results")


def device_info(device: torch.device) -> dict[str, Any]:
    info: dict[str, Any] = {"device": str(device), "torch": torch.__version__, "python": platform.python_version(),
                            "cpu": platform.processor() or platform.machine()}
    if device.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(device)
    else:
        info["threads"] = torch.get_num_threads()
    return info


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class Timer:
    def __init__(self, device: torch.device):
        self.device = device

    def __enter__(self) -> Timer:
        sync(self.device)
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        sync(self.device)
        self.seconds = time.perf_counter() - self.t0


def save(name: str, obj: dict[str, Any]) -> Path:
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(obj, indent=2))
    print(f"[bench] wrote {path}")
    return path
