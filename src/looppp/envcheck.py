"""Environment report for the GPU box (shown as a table in the molab notebook)."""
from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys
from typing import Callable


def _sh(cmd: str, timeout: float = 60) -> tuple[bool, str]:
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return r.returncode == 0, (r.stdout + r.stderr).strip()[-600:]
    except (OSError, subprocess.TimeoutExpired) as e:
        return False, f"{type(e).__name__}: {e}"


_TORCH_PROBE = (
    "import json, torch; ok = torch.cuda.is_available(); "
    "print(json.dumps({'torch': torch.__version__, 'cuda': torch.version.cuda, 'available': ok, "
    "'count': torch.cuda.device_count() if ok else 0, "
    "'name': torch.cuda.get_device_name(0) if ok else '', "
    "'capability': list(torch.cuda.get_device_capability(0)) if ok else None}))"
)


def parse_procfs_information(text: str) -> str:
    """``/proc/driver/nvidia/gpus/<bus>/information`` has a ``Model: <name>`` line."""
    for line in text.splitlines():
        if line.strip().lower().startswith("model:"):
            return line.split(":", 1)[1].strip()
    return ""


def gpu_probe(python: str = sys.executable, sh: Callable[[str, float], tuple[bool, str]] = _sh) -> dict:
    """Find the GPU name through every available route, so a container without
    nvidia-smi (e.g. molab) is not mistaken for a machine without a GPU.

    Returns {"name", "source", "nvidia_smi", "torch", "procfs", "cuda_visible_devices", "device_nodes"}.
    """
    probe: dict = {"name": "", "source": "", "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "(unset)"),
                   "device_nodes": sorted(glob.glob("/dev/nvidia*"))[:8]}

    if shutil.which("nvidia-smi"):
        ok, out = sh("nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit --format=csv,noheader", 60)
        probe["nvidia_smi"] = out
        if ok and out and "failed" not in out.lower():
            probe["name"], probe["source"] = out.splitlines()[0].split(",")[0].strip(), "nvidia-smi"
    else:
        probe["nvidia_smi"] = "not installed"

    ok, out = sh(f"{python} -c \"{_TORCH_PROBE}\"", 180)
    try:
        info = json.loads(out.strip().splitlines()[-1]) if ok and out.strip() else None
    except (json.JSONDecodeError, IndexError):
        info = None
    probe["torch"] = info if info is not None else out
    if not probe["name"] and info and info.get("name"):
        probe["name"], probe["source"] = info["name"], "torch"

    procfs = []
    for path in sorted(glob.glob("/proc/driver/nvidia/gpus/*/information")):
        try:
            with open(path) as f:
                procfs.append(parse_procfs_information(f.read()))
        except OSError:
            continue
    probe["procfs"] = [p for p in procfs if p] or "no /proc/driver/nvidia/gpus entries"
    if not probe["name"] and isinstance(probe["procfs"], list):
        probe["name"], probe["source"] = probe["procfs"][0], "procfs"
    return probe


def gpu_matches(name: str, expected: str | None) -> bool:
    return not expected or expected.upper() in (name or "").upper()


def env_report(python: str = sys.executable, expected_gpu: str | None = "RTX PRO 6000") -> list[dict]:
    rows: list[dict] = []

    def add(check: str, ok: bool, detail: str, required: bool = True):
        rows.append({"check": check, "ok": ok, "required": required, "detail": detail})

    probe = gpu_probe(python)
    detail = (f"{probe['name']} (via {probe['source']})" if probe["name"]
              else f"no GPU found; nvidia-smi: {probe['nvidia_smi']}; /dev/nvidia*: {probe['device_nodes'] or 'none'}")
    add("gpu", bool(probe["name"]) and gpu_matches(probe["name"], expected_gpu), detail)

    t = probe["torch"]
    if isinstance(t, dict):
        add("torch", bool(t.get("available")),
            f"{t['torch']} cuda {t['cuda']} available={t['available']} capability={t['capability']}")
    else:
        add("torch", False, str(t))

    ok, out = _sh(f"{python} -c \"import triton;print(triton.__version__)\"")
    add("triton", ok, out)

    from looppp.cudatk import find_cuda_home

    home = find_cuda_home()
    add("CUDA toolkit / nvcc (CUDA C++ load_inline solutions)", bool(home),
        f"{home[0]} (via {home[1]})" if home else "not found: Triton works; install it in section 2b for CUDA C++",
        required=False)
    cxx = shutil.which("g++") or shutil.which("c++")
    add("host C/C++ compiler (Triton launcher, nvcc host)", bool(shutil.which("gcc") or shutil.which("cc")) and bool(cxx),
        f"gcc={shutil.which('gcc') or shutil.which('cc')} g++={cxx}")

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
