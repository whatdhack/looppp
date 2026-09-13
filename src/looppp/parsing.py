"""Parsers for grader logs and model replies. Every ambiguity fails closed."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from looppp import contract as c

_PEAK_RE = re.compile(r"^peak_fraction:\s*([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)\s*$", re.MULTILINE)
_SHAPE_RE = re.compile(r"^shape=(\d+)\s+solution_peak_fraction=([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)\s*$", re.MULTILINE)
_FAIL_RE = re.compile(r"^FAIL:\s*(.+)$", re.MULTILINE)


@dataclass
class CheckOutcome:
    passed: bool
    fail_stage: str | None = None
    reason: str = ""


@dataclass
class BenchOutcome:
    peak_fraction: float | None
    shape_fractions: list[float] = field(default_factory=list)
    fail_stage: str | None = None
    reason: str = ""


def parse_check(returncode: int, log: str, timed_out: bool = False) -> CheckOutcome:
    """PASS requires exit code 0, a bare ``PASS`` line, and no ``FAIL:`` line."""
    if timed_out:
        return CheckOutcome(False, c.STAGE_TIMEOUT, "check.py timed out")
    fail = _FAIL_RE.search(log)
    if fail:
        reason = fail.group(1).strip()
        low = reason.lower()
        if low.startswith("import error"):
            stage = c.STAGE_IMPORT
        elif low.startswith("forbidden op"):
            stage = c.STAGE_FORBIDDEN
        else:
            stage = c.STAGE_CHECK
        return CheckOutcome(False, stage, reason)
    if returncode != 0:
        last = next((ln for ln in reversed(log.strip().splitlines()) if ln.strip()), "")
        return CheckOutcome(False, c.STAGE_CHECK, f"check.py exited {returncode}: {last[:300]}")
    if not re.search(r"^PASS\s*$", log, re.MULTILINE):
        return CheckOutcome(False, c.STAGE_CHECK, "check.py exited 0 without printing PASS")
    return CheckOutcome(True)


def parse_benchmark(returncode: int, log: str, timed_out: bool = False,
                    expected_shapes: int | None = None) -> BenchOutcome:
    """A score needs exit 0, exactly one line per shape (indices 0..n-1), and exactly one
    final ``peak_fraction:`` line. Extra or missing markers (e.g. printed by the
    candidate itself) make the run unscored."""
    if timed_out:
        return BenchOutcome(None, fail_stage=c.STAGE_TIMEOUT, reason="benchmark.py timed out")
    matches = [(int(m.group(1)), float(m.group(2))) for m in _SHAPE_RE.finditer(log)]
    shapes = [frac for _, frac in matches]
    peaks = _PEAK_RE.findall(log)
    if returncode != 0:
        last = next((ln for ln in reversed(log.strip().splitlines()) if ln.strip()), "")
        return BenchOutcome(None, shapes, c.STAGE_BENCHMARK, f"benchmark.py exited {returncode}: {last[:300]}")
    if not peaks or not shapes:
        return BenchOutcome(None, shapes, c.STAGE_BENCHMARK, "benchmark.py printed no complete score")
    if len(peaks) != 1:
        return BenchOutcome(None, shapes, c.STAGE_BENCHMARK, f"benchmark.py printed {len(peaks)} peak_fraction lines")
    indices = [i for i, _ in matches]
    n = expected_shapes if expected_shapes is not None else len(matches)
    if indices != list(range(n)):
        return BenchOutcome(None, shapes, c.STAGE_BENCHMARK,
                            f"shape markers {indices} do not match the expected 0..{n - 1}")
    return BenchOutcome(float(peaks[0]), shapes)


class ResponseFormatError(ValueError):
    pass


_CODE_RE = re.compile(r"```[ \t]*(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_HYP_RE = re.compile(r"^\s*\**\s*HYPOTHESIS\s*\**\s*:\s*\**\s*(.+?)\s*$", re.MULTILINE | re.IGNORECASE)


def parse_llm_response(text: str) -> tuple[str, str]:
    """Return (hypothesis, code). The code is the last fenced block that defines ``class Model``."""
    blocks = [b for b in _CODE_RE.findall(text or "") if b.strip()]
    if not blocks:
        raise ResponseFormatError("no fenced ```python code block found")
    with_model = [b for b in blocks if re.search(r"^class\s+Model\b", b, re.MULTILINE)]
    code = (with_model or blocks)[-1]
    m = _HYP_RE.search(text)
    hypothesis = m.group(1).strip() if m else ""
    if not hypothesis:
        prose = _CODE_RE.sub("", text).strip().splitlines()
        hypothesis = next((ln.strip() for ln in prose if ln.strip()), "(no hypothesis given)")
    return hypothesis[:300], code.rstrip() + "\n"
