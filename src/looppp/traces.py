"""Rebuild a published solution.py from a KernelBench-Hard HuggingFace trace.

Traces are Claude-Code-format JSONL. The file only exists as a sequence of
Write/Edit tool calls, so we replay them. Used to calibrate a molab GPU against
the published leaderboard.

Caveat: the replayed file is the *last* written version, which is not always
the graded one (an agent may try an experiment after its best result).
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from pathlib import Path

HF_BASE = "https://huggingface.co/datasets/Infatoshi/kernelbench-hard-traces/resolve/main"
HF_PREFIXES = ("", "h100/", "b200/", "rtx3090/")


class TraceReplayError(RuntimeError):
    pass


def download_trace(run_id: str, dest_dir: Path, timeout: float = 120.0) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{run_id}.jsonl"
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    last_err: Exception | None = None
    for prefix in HF_PREFIXES:
        url = f"{HF_BASE}/{prefix}{run_id}.jsonl"
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                dest.write_bytes(r.read())
            return dest
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code != 404:
                break
    raise TraceReplayError(f"trace {run_id} not found on HuggingFace: {last_err}")


def _tool_uses(lines):
    for line in lines:
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        content = (event.get("message") or {}).get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    yield block


def replay_solution(trace_lines, filename: str = "solution.py") -> str:
    """Apply every Write/Edit/MultiEdit on a path ending in *filename*; return the final text."""
    text: str | None = None
    for tu in _tool_uses(trace_lines):
        name = str(tu.get("name", "")).lower()
        args = tu.get("input") or {}
        path = args.get("file_path") or args.get("filePath") or ""
        if not str(path).endswith(filename):
            continue
        if name == "write":
            text = args.get("content", "")
        elif name in ("edit", "multiedit") and text is not None:
            edits = args.get("edits") if name == "multiedit" else [args]
            for e in edits or []:
                old = e.get("old_string", e.get("oldString"))
                new = e.get("new_string", e.get("newString"))
                replace_all = bool(e.get("replace_all") or e.get("replaceAll"))
                if old is None or new is None or old not in text:
                    continue  # the CLI rejected this edit too; the file was unchanged
                text = text.replace(old, new) if replace_all else text.replace(old, new, 1)
    if text is None:
        raise TraceReplayError(f"no Write of {filename} found in trace")
    return text


_COMPILED_EXTENSION = re.compile(r"cpp_extension|load_inline|CUDAExtension|\bnvcc\b|\bcupy\b|nvrtc", re.I)


def needs_cuda_toolkit(code: str) -> bool:
    """True when the solution builds native code at import time (torch cpp_extension, NVRTC, CuPy),
    i.e. it needs nvcc / CUDA headers on the grading box. Pure Triton does not."""
    return bool(_COMPILED_EXTENSION.search(code))


def solution_from_run(run_id: str, cache_dir: Path) -> str:
    path = download_trace(run_id, cache_dir)
    with path.open() as f:
        return replay_solution(f)
