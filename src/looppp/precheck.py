"""Cheap CPU-side checks on WSL so obviously broken candidates never cost a GPU round trip.

This is a filter, not a verdict: check.py on the worker is the only authority.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field

MAX_SOURCE_BYTES = 300_000
_EXIT_CALLS = ("os._exit(", "sys.exit(", "exit(", "quit(")


@dataclass
class PrecheckResult:
    ok: bool
    errors: list[str] = field(default_factory=list)


def _top_level_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update((a.asname or a.name).split(".")[0] for a in node.names)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def precheck(code: str, forbidden: list[str] | None = None) -> PrecheckResult:
    errors: list[str] = []
    if not code or not code.strip():
        return PrecheckResult(False, ["solution.py is empty"])
    if len(code.encode()) > MAX_SOURCE_BYTES:
        errors.append(f"solution.py is {len(code.encode())} bytes (limit {MAX_SOURCE_BYTES})")

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return PrecheckResult(False, [f"SyntaxError at line {e.lineno}: {e.msg}"])

    # check.py / benchmark.py only touch solution.Model (inputs come from reference.py).
    if "Model" not in _top_level_names(tree):
        errors.append("missing top-level `Model` (check.py instantiates solution.Model)")

    # Same rule as KernelBench check.py: a plain substring match on the source.
    for op in forbidden or []:
        if re.search(re.escape(op), code):
            errors.append(f"uses forbidden op `{op}`")

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            src = ast.get_source_segment(code, node.func) or ""
            if any(src + "(" == call for call in _EXIT_CALLS):
                errors.append(f"calls `{src}()` at line {node.lineno}; the grader treats early exits as failures")

    return PrecheckResult(not errors, errors)
