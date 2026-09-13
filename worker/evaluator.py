# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.24.2",
#     "wandb>=0.30.0",
#     "pyyaml>=6.0",
#     "hypothesis>=6.140",
#     "einops>=0.8",
#     "ninja>=1.11",
#     "numpy>=1.26",
# ]
# ///
"""looppp evaluator: open in molab, attach the RTX PRO 6000, connect, (calibrate), start.

Grades candidate kernels from the W&B queue with the pinned KernelBench deck.
The W&B key is typed into a password field and only lives in this session's memory.
"""

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium", app_title="looppp evaluator")


@app.cell
def _():
    import collections
    import importlib.util
    import subprocess
    import sys
    import time
    from pathlib import Path

    import marimo as mo

    return Path, collections, importlib, mo, subprocess, sys, time


@app.cell
def _(Path, importlib, mo, subprocess, sys):
    # Where the looppp code lives: next to this notebook (repo checkout) or a fresh clone.
    LOOPPP_REPO = "https://github.com/whatdhack/looppp.git"  # edit if your repo lives elsewhere

    _here = Path(mo.notebook_dir() or ".").resolve()
    _is_repo = lambda p: (p / "src" / "looppp").is_dir() and (p / "configs" / "run.yaml").is_file()
    repo_root = next((p for p in (_here.parent, _here, Path.home() / "looppp") if _is_repo(p)), None)
    _log = []
    if repo_root is None:
        repo_root = Path.home() / "looppp"
        _r = subprocess.run(["git", "clone", "--depth", "1", LOOPPP_REPO, str(repo_root)],
                            capture_output=True, text=True)
        _log.append(f"git clone {LOOPPP_REPO}: exit {_r.returncode} {_r.stderr.strip()[-300:]}")
    elif (repo_root / ".git").exists():
        _r = subprocess.run(["git", "-C", str(repo_root), "pull", "--ff-only", "-q"], capture_output=True, text=True)
        _log.append(f"git pull: exit {_r.returncode} {_r.stderr.strip()[-300:]}")

    if str(repo_root / "src") not in sys.path:
        sys.path.insert(0, str(repo_root / "src"))

    # Grading dependencies of the KernelBench deck (torch/triton come with molab's image).
    _needed = {"yaml": "pyyaml", "hypothesis": "hypothesis", "einops": "einops", "ninja": "ninja",
               "numpy": "numpy", "wandb": "wandb>=0.30.0"}
    _missing = [pkg for mod, pkg in _needed.items() if importlib.util.find_spec(mod) is None]
    if _missing:
        _r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *_missing], capture_output=True, text=True)
        _log.append(f"pip install {' '.join(_missing)}: exit {_r.returncode} {_r.stderr.strip()[-300:]}")

    mo.vstack([
        mo.md(f"# looppp evaluator\nCode: `{repo_root}`"),
        mo.md("```\n" + ("\n".join(_log) or "code and dependencies already present") + "\n```"),
    ])
    return (repo_root,)


@app.cell
def _(collections, repo_root):
    from looppp.config import load_config
    from looppp.envcheck import env_report, missing_required
    from looppp.grade import KernelBenchGrader
    from looppp.problems import fetch_deck, list_problems
    from looppp.queue.wandb_queue import WandbQueue
    from looppp.traces import solution_from_run
    from looppp.worker import Worker, calibrate, new_worker_id

    cfg = load_config(repo_root)
    holder = {"worker": None, "log": collections.deque(maxlen=300)}  # survives re-runs of downstream cells
    return (
        KernelBenchGrader,
        WandbQueue,
        Worker,
        calibrate,
        cfg,
        env_report,
        fetch_deck,
        holder,
        list_problems,
        missing_required,
        new_worker_id,
        solution_from_run,
    )


@app.cell
def _(cfg, env_report, missing_required, mo, sys):
    env_rows = env_report(python=sys.executable, expected_gpu=cfg.worker.expected_gpu)
    _missing = missing_required(env_rows)
    mo.vstack([
        mo.md("## 1. Environment"),
        mo.callout(mo.md("All required checks passed." if not _missing else
                         f"**Missing:** {', '.join(_missing)}. Attach the GPU (notebook specs button) and re-run."),
                   kind="success" if not _missing else "danger"),
        mo.ui.table(env_rows, selection=None, pagination=False),
    ])
    return


@app.cell
def _(cfg, fetch_deck, list_problems, mo):
    try:
        fetch_deck(cfg.deck.repo, cfg.deck.commit, cfg.deck.subdir, cfg.path(cfg.deck.local_path))
        deck_problems = list_problems(cfg.problems_root)
        _msg = mo.md(f"Pinned deck `{cfg.deck.commit[:10]}`: " + ", ".join(f"`{p}`" for p in deck_problems))
    except Exception as _e:
        deck_problems = []
        _msg = mo.callout(mo.md(f"Deck fetch failed: `{type(_e).__name__}: {_e}`"), kind="danger")
    mo.vstack([mo.md("## 2. KernelBench deck"), _msg])
    return (deck_problems,)


@app.cell
def _(cfg, mo):
    connect_form = mo.ui.dictionary({
        "key": mo.ui.text(kind="password", label="WANDB_API_KEY (use a service-account key)", full_width=True),
        "entity": mo.ui.text(value=cfg.queue.entity or "", label="W&B entity (team)"),
        "project": mo.ui.text(value=cfg.queue.project, label="W&B project"),
    }).form(submit_button_label="Connect", bordered=True)
    mo.vstack([
        mo.md("## 3. Connect to W&B\nThe key stays in this session's memory: it is not written to disk, "
              "not stored in molab secrets, and hidden from graded candidates."),
        connect_form,
    ])
    return (connect_form,)


@app.cell
def _(KernelBenchGrader, WandbQueue, cfg, connect_form, deck_problems, mo, new_worker_id):
    mo.stop(connect_form.value is None, mo.md("Fill in the form and press **Connect**."))
    _v = connect_form.value
    mo.stop(not _v["key"].strip() or not _v["entity"].strip(),
            mo.callout(mo.md("Key and entity are required."), kind="warn"))
    mo.stop(not deck_problems, mo.callout(mo.md("The deck is not available (section 2)."), kind="danger"))
    try:
        queue = WandbQueue(_v["entity"].strip(), _v["project"].strip(), api_key=_v["key"].strip())
        _viewer = queue.api.viewer
        _user = getattr(_viewer, "username", None) or getattr(_viewer, "name", "?")
    except Exception as _e:  # never echo the exception text: it could contain request details
        mo.stop(True, mo.callout(mo.md(f"W&B rejected the connection (`{type(_e).__name__}`)."), kind="danger"))
    worker_id = new_worker_id("molab")
    try:
        grader = KernelBenchGrader(cfg.path(cfg.deck.local_path), cfg.deck.subdir, cfg.deck.problems_dir,
                                   cfg.deck.commit, cfg.worker.expected_gpu, cfg.worker.check_timeout_seconds,
                                   cfg.worker.bench_timeout_seconds, worker_id=worker_id)
    except Exception as _e:
        mo.stop(True, mo.callout(mo.md(f"Grader not ready: `{_e}`"), kind="danger"))
    mo.callout(mo.md(f"Connected as **{_user}** to `{queue.entity}/{queue.project}`  \n"
                     f"GPU `{grader.gpu}` · torch `{grader.torch}` · worker `{worker_id}`"), kind="success")
    return grader, queue, worker_id


@app.cell
def _(cfg, deck_problems, mo):
    _choices = [p for p in cfg.calibration if p in deck_problems]
    calib_form = mo.ui.dictionary({
        "problem": mo.ui.dropdown(_choices, value=_choices[0] if _choices else None, label="problem"),
        "runs": mo.ui.slider(1, 5, value=3, show_value=True, label="runs"),
    }).form(submit_button_label="Run calibration")
    mo.vstack([
        mo.md("## 4. Calibrate this GPU (recommended once per session)\nRebuilds a published KernelBench "
              "solution from its HuggingFace trace and grades it here. A ratio well below 1.0 means this GPU is "
              "slower than KernelBench's; looppp scores then compare with each other, not with the leaderboard."),
        calib_form,
    ])
    return (calib_form,)


@app.cell
def _(calib_form, calibrate, cfg, grader, holder, mo, queue, solution_from_run, worker_id):
    mo.stop(calib_form.value is None)
    mo.stop(holder["worker"] is not None and holder["worker"].is_running,
            mo.callout(mo.md("Stop the worker first: one GPU job at a time keeps timings clean."), kind="warn"))
    _p = calib_form.value["problem"]
    _t = cfg.calibration[_p]
    with mo.status.spinner(title=f"Calibrating {_p}: {calib_form.value['runs']} gradings..."):
        _code = solution_from_run(_t.run_id, cfg.path(".looppp/traces"))
        calib_report = calibrate(grader, _p, _code, calib_form.value["runs"], _t.published_peak_fraction)
    queue.worker_log(worker_id, {f"calibration/{_p}": calib_report})
    mo.callout(mo.md(f"```\n{calib_report}\n```"),
               kind="success" if calib_report.get("scored") else "danger")
    return


@app.cell
def _(mo):
    start_btn = mo.ui.run_button(label="Start worker", kind="success")
    stop_btn = mo.ui.run_button(label="Stop worker", kind="danger")
    mo.vstack([mo.md("## 5. Worker\nKeep this tab open. Stopping finishes the current candidate first."),
               mo.hstack([start_btn, stop_btn], justify="start")])
    return start_btn, stop_btn


@app.cell
def _(Worker, cfg, grader, holder, mo, queue, start_btn, time):
    mo.stop(not start_btn.value)
    if holder["worker"] is not None and holder["worker"].is_running:
        _out = mo.callout(mo.md("Worker already running."), kind="info")
    else:
        holder["worker"] = Worker(
            queue, grader, poll_seconds=cfg.worker.poll_seconds, heartbeat_seconds=cfg.worker.heartbeat_seconds,
            max_attempts=cfg.queue.max_attempts,
            log=lambda s: holder["log"].append(f"{time.strftime('%H:%M:%S')} {s}"),
        )
        holder["worker"].start_thread()
        _out = mo.callout(mo.md("Worker started."), kind="success")
    _out
    return


@app.cell
def _(holder, mo, stop_btn):
    mo.stop(not stop_btn.value)
    if holder["worker"] is not None:
        holder["worker"].stop()
    mo.md("Stop requested." if holder["worker"] is not None else "No worker running.")
    return


@app.cell
def _(mo):
    refresh = mo.ui.refresh(options=["10s", "30s", "60s"], default_interval="30s", label="auto-refresh")
    refresh
    return (refresh,)


@app.cell
def _(holder, mo, queue, refresh):
    refresh.value  # re-run on every tick
    _w = holder["worker"]
    _st = _w.status if _w is not None else {"state": "not started", "graded": 0, "errors": 0, "current": "", "last": None}
    try:
        _rows = [{
            "status": r.status,
            "peak_fraction": r.result.peak_fraction if r.result else None,
            "fail_stage": r.result.fail_stage if r.result else None,
            "problem": r.spec.problem, "gen": f"{r.spec.generation}.{r.spec.index}",
            "model": r.spec.model, "hypothesis": r.spec.hypothesis[:90],
        } for r in queue.list_candidates(limit=15)]
    except Exception as _e:
        _rows = [{"error": f"{type(_e).__name__}"}]
    _last = (_st.get("last") or {}).get("peak_fraction")
    mo.vstack([
        mo.hstack([mo.stat(str(_st["state"]), label="worker"), mo.stat(_st["graded"], label="graded"),
                   mo.stat(_st["errors"], label="errors"),
                   mo.stat(f"{_last:.4f}" if _last is not None else "-", label="last peak_fraction")]),
        mo.md(f"Current: `{_st.get('current') or '-'}`"),
        mo.ui.table(_rows, selection=None, pagination=False),
        mo.accordion({"Worker log": mo.md("```\n" + "\n".join(list(holder["log"])[-40:]) + "\n```")}),
    ])
    return


if __name__ == "__main__":
    app.run()
