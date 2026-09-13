from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path

import pytest

from looppp.config import load_config
from looppp.problems import Problem

TRUSTED_ENTRYPOINT = '''
import runpy, sys
from pathlib import Path

def main():
    script = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(script.parent))
    try:
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit as exc:
        if exc.code not in (None, 0):
            raise
        print("FAIL: exited successfully before normal completion", file=sys.stderr)
        return 1
    return 0

raise SystemExit(main())
'''

CHECK = '''
import os, re, sys
from pathlib import Path
import yaml
if os.environ.get("WANDB_API_KEY") or os.environ.get("MY_SECRET_TOKEN"):
    print("LEAK: secret visible to grading subprocess")
try:
    import solution
except Exception as e:
    print(f"FAIL: import error: {e}")
    sys.exit(1)
meta = yaml.safe_load(Path("problem.yaml").read_text())
src = Path("solution.py").read_text()
for op in meta.get("forbidden", []):
    if re.search(re.escape(op), src):
        print(f"FAIL: forbidden op used: {op}")
        sys.exit(1)
if solution.Model().forward(3) != 6:
    print("FAIL: shape 0 seed 42: wrong output")
    sys.exit(1)
print("PASS")
'''

BENCH = '''
import math
import shapes, solution
fr = []
for i, _ in enumerate(shapes.SHAPES):
    f = getattr(solution, "SCORE", 0.1)
    fr.append(f)
    print(f"shape={i} variant=solution tflops=1.0 gbps=1.0 ms=1.0")
    print(f"shape={i} solution_peak_fraction={f:.4f}")
g = math.exp(sum(math.log(x) for x in fr) / len(fr))
print(f"peak_fraction: {g:.4f}")
'''

REFERENCE = '''
class Model:
    def forward(self, x):
        return x + x

def get_inputs():
    return [3]

def get_init_inputs():
    return []
'''


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def fake_deck(tmp_path: Path) -> dict:
    repo = tmp_path / "deck"
    hard = repo / "benchmarks" / "hard"
    (hard / "src" / "eval").mkdir(parents=True)
    (hard / "src" / "eval" / "trusted_entrypoint.py").write_text(TRUSTED_ENTRYPOINT)
    p = hard / "problems-rtxpro6000" / "01_toy"
    p.mkdir(parents=True)
    (p / "check.py").write_text(CHECK)
    (p / "benchmark.py").write_text(BENCH)
    (p / "shapes.py").write_text("SHAPES = [{'M': 1}, {'M': 2}, {'M': 3}]\n")
    (p / "reference.py").write_text(REFERENCE)
    (p / "PROMPT.txt").write_text("Write a fast doubling kernel in solution.py.\n")
    (p / "problem.yaml").write_text("name: 01_toy\nforbidden:\n  - torch.nn.functional.linear\n")
    _git(repo, "init", "-q")
    _git(repo, "add", ".")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "deck")
    return {"repo": repo, "subdir": "benchmarks/hard", "problems_dir": "problems-rtxpro6000",
            "commit": _git(repo, "rev-parse", "HEAD"), "problem_dir": p}


@pytest.fixture
def toy_problem(fake_deck) -> Problem:
    from looppp.problems import load_problem

    return load_problem(fake_deck["problem_dir"].parent, "01_toy")


@pytest.fixture
def cfg(tmp_path: Path):
    root = tmp_path / "repo"
    (root / "configs").mkdir(parents=True)
    (root / "configs" / "run.yaml").write_text(textwrap.dedent("""
        loop:
          problem: 01_toy
          model: stub
          candidates_per_generation: 2
          max_generations: 3
          patience: 10
          wall_budget_hours: 1
          state_dir: state
        queue:
          backend: local
          local_root: queue
          poll_seconds: 0.02
          wait_timeout_hours: 1
          max_worker_down_hours: 1
          heartbeat_stale_seconds: 5
        archive:
          dir: archive
    """))
    return load_config(root)


GOOD_SOLUTION = '''
SCORE = {score}

class Model:
    def forward(self, x):
        return x * 2
'''


def reply(hypothesis: str, code: str) -> str:
    return f"HYPOTHESIS: {hypothesis}\n\n```python\n{code.strip()}\n```\n"
