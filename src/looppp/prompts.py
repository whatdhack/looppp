"""Prompt construction for the proposer."""
from __future__ import annotations

from dataclasses import dataclass

from looppp.contract import GradeResult
from looppp.problems import Problem

SYSTEM = """\
You are an expert GPU kernel engineer (CUDA, Triton, CUTLASS, PyTorch internals) iterating on a
kernel with a remote evaluator.

How this works:
- You cannot run commands. Each reply you write is ONE complete candidate solution.py.
- An evaluator on the target GPU restores the problem directory, drops in your solution.py, runs
  `python check.py` (correctness on every shape, several seeds, numeric stress) and, only if it
  prints PASS, `python benchmark.py` (score = peak_fraction: geometric mean over shapes of the
  fraction of the GPU's hardware ceiling). You then get the logs back.
- solution.py must define `Model` with the same constructor and forward semantics as reference.py
  (check.py loads reference.Model's state_dict into solution.Model). Inputs come from reference.py.
- Never use the forbidden ops. Never call sys.exit/os._exit. Do not print lines starting with
  `PASS`, `FAIL:`, `shape=` or `peak_fraction:` yourself.
- Anything compiled must build inside the evaluator (Triton JIT, torch.utils.cpp_extension.load_inline).
  A kernel that fails to compile scores nothing, so prefer approaches you are confident will build.

Reply format (exactly):
HYPOTHESIS: <one line: the specific change you are testing and why it should be faster or fix the failure>

```python
<the complete solution.py>
```
"""


@dataclass
class HistoryItem:
    generation: int
    index: int
    hypothesis: str
    status: str
    result: GradeResult | None


def _fmt_result(r: GradeResult | None, status: str) -> str:
    if r is None:
        return status
    if r.scored:
        shapes = ", ".join(f"{x:.3f}" for x in r.shape_fractions)
        return f"PASS peak_fraction={r.peak_fraction:.4f} (per shape: {shapes})"
    if r.correct:
        return f"PASS but unscored ({r.fail_stage}: {r.fail_reason})"
    return f"FAILED at {r.fail_stage}: {r.fail_reason}"


def history_table(items: list[HistoryItem]) -> str:
    if not items:
        return "(no graded attempts yet)"
    lines = [f"- gen {h.generation}.{h.index}: {h.hypothesis}\n    -> {_fmt_result(h.result, h.status)}" for h in items]
    return "\n".join(lines)


MODE_EXPLOIT = "exploit"
MODE_EXPLORE = "explore"
MODE_SEED = "seed"


def build_propose_messages(problem: Problem, *, parent_code: str, parent_label: str,
                           parent_result: GradeResult | None, history: list[HistoryItem],
                           failures: list[tuple[HistoryItem, str]], sibling_hypotheses: list[str],
                           index: int, n: int, best_peak: float | None, mode: str = MODE_EXPLOIT,
                           tried: list[str] | None = None, stagnant_generations: int = 0,
                           min_rel_improvement: float = 0.05) -> list[dict]:
    forbidden = "\n".join(f"- {op}" for op in problem.forbidden) or "- (none listed)"
    parts = [
        "## Task (as given to engineers on this problem)\n" + problem.prompt.strip(),
        "## Forbidden ops (checked by substring match on solution.py)\n" + forbidden,
        "## reference.py\n```python\n" + problem.reference.strip() + "\n```",
        "## shapes.py\n```python\n" + problem.shapes.strip() + "\n```",
        f"## Current parent: {parent_label}\n" + _fmt_result(parent_result, "not graded")
        + "\n```python\n" + parent_code.strip() + "\n```",
        f"## Best graded peak_fraction so far: {best_peak:.4f}" if best_peak is not None
        else "## Best graded peak_fraction so far: none yet (first priority: pass check.py)",
        "## Recent attempts (newest last)\n" + history_table(history),
    ]
    for item, log in failures:
        parts.append(f"## Evaluator log for failed attempt gen {item.generation}.{item.index}\n```\n{log.strip()}\n```")
    if mode == MODE_EXPLORE:
        ask = (f"## Your job: EXPLORE (candidate {index + 1} of {n})\n"
               f"The best score has not improved by {min_rel_improvement:.0%} for {stagnant_generations} generations, "
               "so parameter tweaks of the current design are exhausted. Write a STRUCTURALLY DIFFERENT kernel: "
               "a different algorithm or data layout, a different decomposition per shape regime (decode-sized vs "
               "large batches), a different precision / accumulation strategy, or a different backend (Triton vs a "
               "C++/CUDA extension built with torch.utils.cpp_extension.load_inline). Changing only block sizes, "
               "num_warps, num_stages or autotune lists does NOT count. You may start from reference.py instead "
               "of the parent. It must still pass check.py. Think about which shape has the lowest per-shape "
               "fraction and what actually bounds it before choosing.")
        if tried:
            ask += "\n\nApproaches already tried in this loop (do not repeat them):\n" + "\n".join(f"- {t}" for t in tried)
    else:
        ask = (f"## Your job\nWrite candidate {index + 1} of {n} for this generation. Start from the parent, "
               "fix any failure first, then make ONE focused improvement.")
    if sibling_hypotheses:
        ask += ("\nOther candidates in this generation already test these ideas; pick a DIFFERENT one:\n"
                + "\n".join(f"- {h}" for h in sibling_hypotheses))
    parts.append(ask)
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n\n".join(parts)}]


def build_repair_messages(previous: list[dict], reply_text: str, errors: list[str]) -> list[dict]:
    msg = ("Your candidate was rejected by local pre-checks before reaching the GPU:\n"
           + "\n".join(f"- {e}" for e in errors)
           + "\n\nReply again in the exact format (HYPOTHESIS line + one complete ```python block).")
    return [*previous, {"role": "assistant", "content": reply_text}, {"role": "user", "content": msg}]
