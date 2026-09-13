"""Worker-side grading. Runs on the GPU box (molab); never imported by the agent loop.

The candidate is untrusted LLM-written code, so:
  * check.py / benchmark.py run in a fresh subprocess group with a timeout (a CUDA
    fault or a hung compile only kills that process group, not the worker);
  * the subprocess environment has every secret-looking variable removed;
  * the whole deck subdir is restored from git before and after every candidate,
    and any change to tracked grader files (outside solution.py) fails the run.
"""
from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from looppp import contract as c
from looppp.contract import GradeResult
from looppp.envcheck import gpu_matches, gpu_probe
from looppp.parsing import parse_benchmark, parse_check
from looppp.problems import count_shapes, deck_head

SENSITIVE_PREFIXES = ("WANDB_", "GITHUB_", "GH_", "GIT_ASKPASS", "HF_", "HUGGING_FACE", "OPENAI_",
                      "ANTHROPIC_", "MARIMO_", "AWS_", "GOOGLE_", "AZURE_", "OPENROUTER_")
SENSITIVE_SUBSTRINGS = ("TOKEN", "SECRET", "PASSWORD", "PASSWD", "API_KEY", "APIKEY", "CREDENTIAL", "PRIVATE_KEY")


def scrubbed_env(base: dict[str, str] | None = None, extra: dict[str, str] | None = None) -> dict[str, str]:
    base = dict(os.environ if base is None else base)
    env = {k: v for k, v in base.items()
           if not k.upper().startswith(SENSITIVE_PREFIXES)
           and not any(s in k.upper() for s in SENSITIVE_SUBSTRINGS)}
    env.update(extra or {})
    return env


@dataclass
class ProcResult:
    returncode: int
    output: str
    timed_out: bool
    seconds: float


def _kill_group(pid: int, grace: float) -> None:
    for sig, wait in ((signal.SIGTERM, grace), (signal.SIGKILL, 0.0)):
        try:
            os.killpg(pid, sig)
        except (ProcessLookupError, PermissionError):
            return
        if wait:
            time.sleep(wait)


def run_process(cmd: list[str], cwd: Path, env: dict[str, str], timeout: float) -> ProcResult:
    """Run *cmd* in its own session; on timeout kill the whole group. Output goes to a temp
    file (not a pipe) so a chatty child can never deadlock the worker."""
    start = time.monotonic()
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace") as out:
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=out, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, start_new_session=True)
        timed_out = False
        try:
            rc = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_group(proc.pid, grace=5.0)
            rc = proc.wait()
        _kill_group(proc.pid, grace=0.0)  # reap stray children (compile workers) left behind
        out.seek(0)
        return ProcResult(rc, out.read(), timed_out, time.monotonic() - start)


class GpuCheckError(RuntimeError):
    """The GPU could not be identified or is not the expected one. ``probe`` holds the evidence."""

    def __init__(self, message: str, probe: dict):
        super().__init__(message)
        self.probe = probe


class Grader(Protocol):
    worker_id: str

    def grade(self, problem: str, code: str) -> GradeResult: ...

    def describe(self) -> dict: ...


class KernelBenchGrader:
    def __init__(self, deck_repo: Path, deck_subdir: str, problems_dir: str, expected_commit: str,
                 expected_gpu: str | None, check_timeout: float, bench_timeout: float,
                 worker_id: str = "", python: str = sys.executable):
        self.repo = Path(deck_repo)
        self.subdir = deck_subdir
        self.deck_root = self.repo / deck_subdir
        self.problems_root = self.deck_root / problems_dir
        self.entrypoint = self.deck_root / "src" / "eval" / "trusted_entrypoint.py"
        self.expected_commit = expected_commit
        self.check_timeout, self.bench_timeout = check_timeout, bench_timeout
        self.worker_id, self.python = worker_id, python

        head = deck_head(self.repo)
        if head != expected_commit:
            raise RuntimeError(f"deck at {head}, expected {expected_commit}; run fetch_deck first")
        if not self.entrypoint.is_file():
            raise RuntimeError(f"missing {self.entrypoint}")
        self.probe = gpu_probe(python)
        self.gpu = self.probe["name"]
        if expected_gpu and not self.gpu:
            raise GpuCheckError(f"no GPU found (expected {expected_gpu!r})", self.probe)
        if not gpu_matches(self.gpu, expected_gpu):
            raise GpuCheckError(f"GPU {self.gpu!r} does not match expected {expected_gpu!r}", self.probe)
        t = self.probe.get("torch")
        self.torch = f"{t['torch']} (cuda {t['cuda']})" if isinstance(t, dict) else ""

    def describe(self) -> dict:
        return {"gpu_name": self.gpu, "gpu_source": self.probe.get("source", ""), "torch_version": self.torch,
                "deck_commit": self.expected_commit}

    def _git(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["git", "-C", str(self.repo), *args], capture_output=True, text=True)

    def _restore(self) -> None:
        self._git("checkout", "--", self.subdir)
        self._git("clean", "-fdxq", "--", self.subdir)

    def _tampered(self, problem: str) -> bool:
        sol = f"{self.subdir}/{self.problems_root.name}/{problem}/{c.SOLUTION_FILENAME}"
        r = self._git("diff", "--quiet", "HEAD", "--", self.subdir, f":(exclude){sol}")
        return r.returncode != 0

    def _result(self, t0: float, **kw) -> GradeResult:
        return GradeResult(grade_seconds=round(time.monotonic() - t0, 1), gpu_name=self.gpu,
                           torch_version=self.torch, worker_id=self.worker_id, **kw)

    def grade(self, problem: str, code: str) -> GradeResult:
        t0 = time.monotonic()
        pdir = self.problems_root / problem
        if not re.fullmatch(r"[A-Za-z0-9_\-]+", problem) or not (pdir / "check.py").is_file():
            return self._result(t0, correct=False, fail_stage=c.STAGE_INTEGRITY, fail_reason=f"unknown problem {problem!r}")
        n_shapes = count_shapes((pdir / "shapes.py").read_text()) if (pdir / "shapes.py").exists() else None

        self._restore()
        try:
            (pdir / c.SOLUTION_FILENAME).write_text(code)
            with tempfile.TemporaryDirectory(prefix="looppp-cache-") as cache:
                env = scrubbed_env(extra={
                    "TRITON_CACHE_DIR": f"{cache}/triton", "TORCH_EXTENSIONS_DIR": f"{cache}/torch_extensions",
                    "CUDA_CACHE_PATH": f"{cache}/nv", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
                })
                chk = run_process([self.python, str(self.entrypoint), "check.py"], pdir, env, self.check_timeout)
                check_tail = c.tail(chk.output)
                if self._tampered(problem):
                    return self._result(t0, correct=False, fail_stage=c.STAGE_TAMPER, check_tail=check_tail,
                                        fail_reason="grader files changed while check.py ran")
                co = parse_check(chk.returncode, chk.output, chk.timed_out)
                if not co.passed:
                    return self._result(t0, correct=False, fail_stage=co.fail_stage, fail_reason=co.reason,
                                        check_tail=check_tail)

                bench = run_process([self.python, str(self.entrypoint), "benchmark.py"], pdir, env, self.bench_timeout)
                bench_tail = c.tail(bench.output)
                if self._tampered(problem):
                    return self._result(t0, correct=False, fail_stage=c.STAGE_TAMPER, check_tail=check_tail,
                                        bench_tail=bench_tail, fail_reason="grader files changed while benchmark.py ran")
                bo = parse_benchmark(bench.returncode, bench.output, bench.timed_out, expected_shapes=n_shapes)
                return self._result(t0, correct=True, peak_fraction=bo.peak_fraction,
                                    shape_fractions=bo.shape_fractions, fail_stage=bo.fail_stage,
                                    fail_reason=bo.reason, check_tail=check_tail, bench_tail=bench_tail)
        finally:
            self._restore()


class FakeGrader:
    """CPU stand-in with the same contract, for dry runs of the whole loop.

    Score is derived from the code hash (0.05-0.35) unless the code contains
    ``FAKE_SCORE=<float>``; ``FAKE_FAIL`` makes check fail.
    """

    def __init__(self, worker_id: str = "fake-worker", delay_seconds: float = 0.0):
        self.worker_id, self.delay = worker_id, delay_seconds

    def describe(self) -> dict:
        return {"gpu_name": "FAKE", "torch_version": "", "deck_commit": "fake"}

    def grade(self, problem: str, code: str) -> GradeResult:
        if self.delay:
            time.sleep(self.delay)
        base = dict(gpu_name="FAKE", worker_id=self.worker_id, grade_seconds=self.delay)
        if "FAKE_FAIL" in code:
            return GradeResult(correct=False, fail_stage=c.STAGE_CHECK, check_tail="FAIL: fake failure",
                               fail_reason="fake failure", **base)
        m = re.search(r"FAKE_SCORE\s*=\s*([0-9.]+)", code)
        score = float(m.group(1)) if m else 0.05 + (int(c.sha256_text(code)[:6], 16) % 300) / 1000
        return GradeResult(correct=True, peak_fraction=round(score, 4), shape_fractions=[round(score, 4)] * 5,
                           check_tail="PASS", bench_tail=f"peak_fraction: {score:.4f}", **base)
