"""Environment report for the GPU box (shown as a table in the molab notebook)."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys


def _sh(cmd: str, timeout: float = 60) -> tuple[bool, str]:
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return r.returncode == 0, (r.stdout + r.stderr).strip()[-600:]
    except subprocess.TimeoutExpired:
        return False, f"timed out after {timeout}s"


def env_report(python: str = sys.executable, expected_gpu: str | None = "RTX PRO 6000") -> list[dict]:
    rows: list[dict] = []

    def add(check: str, ok: bool, detail: str, required: bool = True):
        rows.append({"check": check, "ok": ok, "required": required, "detail": detail})

    ok, out = _sh("nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit --format=csv,noheader")
    add("gpu", ok and (not expected_gpu or expected_gpu.upper() in out.upper()), out or "nvidia-smi not found")

    ok, out = _sh(f"{python} -c \"import torch;print(torch.__version__, 'cuda', torch.version.cuda, "
                  f"'available', torch.cuda.is_available(), 'capability', "
                  f"torch.cuda.get_device_capability() if torch.cuda.is_available() else None)\"", 180)
    add("torch", ok and "available True" in out, out)

    ok, out = _sh(f"{python} -c \"import triton;print(triton.__version__)\"")
    add("triton", ok, out)

    nvcc = shutil.which("nvcc") or next((p for p in ("/usr/local/cuda/bin/nvcc",) if os.path.exists(p)), None)
    add("nvcc (CUDA C++/CUTLASS solutions)", bool(nvcc), nvcc or "not found: Triton-only solutions will work",
        required=False)

    for mod in ("yaml", "hypothesis", "einops", "numpy", "ninja"):
        ok, out = _sh(f"{python} -c \"import {mod}\"")
        add(f"python module {mod}", ok, "ok" if ok else out)

    add("git", bool(shutil.which("git")), shutil.which("git") or "missing")
    ok, out = _sh("curl -s -o /dev/null -w '%{http_code}' https://github.com", 30)
    add("internet: github.com", ok and out.startswith(("2", "3")), out)
    ok, out = _sh("curl -s -o /dev/null -w '%{http_code}' https://api.wandb.ai/healthz", 30)
    add("internet: api.wandb.ai", ok and out.startswith(("2", "3", "4")), out)
    ok, out = _sh("curl -s -o /dev/null -w '%{http_code}' https://huggingface.co", 30)
    add("internet: huggingface.co (calibration traces)", ok and out.startswith(("2", "3")), out, required=False)

    ok, out = _sh("df -h . | tail -1")
    add("disk", ok, out, required=False)
    add("cpus", True, str(os.cpu_count()), required=False)
    return rows


def missing_required(rows: list[dict]) -> list[str]:
    return [r["check"] for r in rows if r["required"] and not r["ok"]]
