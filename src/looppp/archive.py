"""Best graded solution per problem: archive/<problem>/best.py + best.json."""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path


class Archive:
    def __init__(self, root: Path, git_commit: bool = False):
        self.root, self.git_commit = Path(root), git_commit

    def _dir(self, problem: str) -> Path:
        return self.root / problem

    def best(self, problem: str) -> tuple[str, dict] | None:
        d = self._dir(problem)
        if not (d / "best.py").is_file() or not (d / "best.json").is_file():
            return None
        return (d / "best.py").read_text(), json.loads((d / "best.json").read_text())

    def best_peak(self, problem: str) -> float | None:
        cur = self.best(problem)
        return float(cur[1]["peak_fraction"]) if cur else None

    def maybe_update(self, problem: str, code: str, meta: dict) -> bool:
        """Store if strictly better than the archived best. Returns True when updated."""
        peak = meta.get("peak_fraction")
        if peak is None:
            return False
        current = self.best_peak(problem)
        if current is not None and peak <= current:
            return False
        d = self._dir(problem)
        d.mkdir(parents=True, exist_ok=True)
        (d / "best.py").write_text(code)
        (d / "best.json").write_text(json.dumps({**meta, "archived_ts": time.time()}, indent=2, default=str) + "\n")
        if self.git_commit:
            self._commit(problem, peak, current)
        return True

    def _commit(self, problem: str, peak: float, previous: float | None) -> None:
        d = self._dir(problem)
        top = subprocess.run(["git", "-C", str(d), "rev-parse", "--show-toplevel"], capture_output=True, text=True)
        if top.returncode != 0:
            print(f"[looppp] archive not in a git repo; skipping commit ({top.stderr.strip()})")
            return
        rel = d.resolve().relative_to(Path(top.stdout.strip()).resolve())
        prev = f"{previous:.4f}" if previous is not None else "none"
        msg = f"archive: {problem} peak_fraction {prev} -> {peak:.4f}"
        subprocess.run(["git", "-C", top.stdout.strip(), "add", "--", str(rel)], check=True)
        # `git commit -- <path>` commits only the archive files, never unrelated staged work.
        r = subprocess.run(["git", "-C", top.stdout.strip(), "commit", "-q", "-m", msg, "--", str(rel)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(f"[looppp] archive commit failed: {r.stderr.strip() or r.stdout.strip()}")
