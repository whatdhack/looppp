"""KernelBench problem deck: pinned sparse clone + problem loading."""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Problem:
    name: str
    dir: Path
    prompt: str
    reference: str
    shapes: str
    meta: dict[str, Any]

    @property
    def forbidden(self) -> list[str]:
        return list(self.meta.get("forbidden") or [])


def load_problem(problems_root: Path, name: str) -> Problem:
    d = problems_root / name
    if not (d / "PROMPT.txt").is_file():
        raise FileNotFoundError(f"problem {name!r} not found under {problems_root} (run `looppp fetch-deck`)")
    return Problem(
        name=name,
        dir=d,
        prompt=(d / "PROMPT.txt").read_text(),
        reference=(d / "reference.py").read_text(),
        shapes=(d / "shapes.py").read_text() if (d / "shapes.py").exists() else "",
        meta=yaml.safe_load((d / "problem.yaml").read_text()) or {},
    )


def count_shapes(shapes_src: str) -> int | None:
    """Number of entries in ``SHAPES = [...]`` (None if it is not a plain literal)."""
    import ast

    try:
        tree = ast.parse(shapes_src)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "SHAPES" for t in node.targets):
            try:
                return len(ast.literal_eval(node.value))
            except ValueError:
                return None
    return None


def list_problems(problems_root: Path) -> list[str]:
    if not problems_root.is_dir():
        return []
    return sorted(p.name for p in problems_root.iterdir() if (p / "PROMPT.txt").is_file())


def _git(dest: Path, *args: str) -> str:
    r = subprocess.run(["git", "-C", str(dest), *args], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {r.stderr.strip()[-500:]}")
    return r.stdout.strip()


def deck_head(dest: Path) -> str | None:
    if not (dest / ".git").exists():
        return None
    try:
        return _git(dest, "rev-parse", "HEAD")
    except RuntimeError:
        return None


def fetch_deck(repo: str, commit: str, subdir: str, dest: Path) -> Path:
    """Shallow sparse checkout of *subdir* at *commit* into *dest*. Idempotent."""
    dest = Path(dest)
    if deck_head(dest) == commit:
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    if not (dest / ".git").exists():
        _git(dest, "init", "-q")
        _git(dest, "remote", "add", "origin", repo)
    _git(dest, "sparse-checkout", "set", subdir)
    _git(dest, "fetch", "-q", "--depth", "1", "origin", commit)
    _git(dest, "checkout", "-q", "--detach", "FETCH_HEAD")
    head = deck_head(dest)
    if head != commit:
        raise RuntimeError(f"deck checkout is at {head}, expected {commit}")
    return dest
