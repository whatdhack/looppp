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
    from looppp.cudatk import ensure_cuda_toolkit
    from looppp.envcheck import env_report, gpu_probe, missing_required
    from looppp.grade import GpuCheckError, KernelBenchGrader
    from looppp.problems import fetch_deck, list_problems
    from looppp.queue.wandb_queue import WandbQueue, explain_wandb_error
    from looppp.traces import calibration_solution, needs_cuda_toolkit
    from looppp.worker import Worker, calibrate, new_worker_id

    cfg = load_config(repo_root)
    holder = {"worker": None, "log": collections.deque(maxlen=300)}  # survives re-runs of downstream cells
    return (
        GpuCheckError,
        KernelBenchGrader,
        WandbQueue,
        Worker,
        calibrate,
        calibration_solution,
        cfg,
        ensure_cuda_toolkit,
        env_report,
        explain_wandb_error,
        fetch_deck,
        gpu_probe,
        holder,
        list_problems,
        missing_required,
        needs_cuda_toolkit,
        new_worker_id,
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
def _(mo):
    cuda_btn = mo.ui.run_button(label="Re-check / retry CUDA toolkit install", kind="neutral")
    cuda_btn
    return (cuda_btn,)


@app.cell
def _(cfg, cuda_btn, ensure_cuda_toolkit, holder, mo, sys):
    # Most strong KernelBench kernels build a C++/CUDA extension, so the toolkit is installed automatically
    # on load (compiler wheels only, ~60 MB, torch's own CUDA libraries untouched). The button re-runs this
    # cell. The grader looks the toolkit up at every grading, so no other cell has to re-run.
    cuda_btn.value
    with mo.status.spinner(title="Checking / installing CUDA toolkit (nvcc + headers)..."):
        cuda_report = ensure_cuda_toolkit(sys.executable, cfg.path(".looppp"), install=True)
    holder["cuda_report"] = cuda_report
    if cuda_report.ok:
        _msg = mo.callout(mo.md(f"**CUDA toolkit ready:** `{cuda_report.cuda_home}` · nvcc {cuda_report.nvcc_version} "
                                f"(via {cuda_report.source}) · torch CUDA {cuda_report.torch_cuda}  \n"
                                "CUDA C++ (`load_inline`) and Triton solutions can both be graded."), kind="success")
    else:
        _msg = mo.callout(mo.md(
            "**CUDA toolkit install failed.** Triton solutions still grade; C++/CUDA extension solutions will "
            f"fail with `toolchain`. torch CUDA: {cuda_report.torch_cuda or '?'}. Log below; press the button "
            "above to retry."
            + ("\n\n```\n" + "\n".join(cuda_report.log)[-1500:] + "\n```" if cuda_report.log else "")),
            kind="danger")
    mo.vstack([mo.md("## 2b. CUDA toolkit (for C++/CUDA extension solutions)"), _msg])
    return


@app.cell
def _(cfg, mo):
    connect_form = mo.ui.dictionary({
        "key": mo.ui.text(kind="password", label="WANDB_API_KEY (use a service-account key)", full_width=True),
        "entity": mo.ui.text(value=cfg.queue.entity or "", label="W&B entity (team)"),
        "project": mo.ui.text(value=cfg.queue.project, label="W&B project"),
        "expected_gpu": mo.ui.text(value=cfg.worker.expected_gpu or "", label="expected GPU (substring of its name)"),
        "allow_unverified_gpu": mo.ui.checkbox(
            value=False, label="grade even if the GPU cannot be identified (results are tagged with the probe)"),
    }).form(submit_button_label="Connect", bordered=True)
    mo.vstack([
        mo.md("## 3. Connect to W&B\nThe key stays in this session's memory: it is not written to disk, "
              "not stored in molab secrets, and hidden from graded candidates."),
        connect_form,
    ])
    return (connect_form,)


@app.cell
def _(GpuCheckError, KernelBenchGrader, WandbQueue, cfg, connect_form, deck_problems, explain_wandb_error,
      gpu_probe, mo, new_worker_id, sys):
    # This cell never calls mo.stop: when not connected it exports queue/grader = None and shows why, so the
    # cells below can say "connect first" instead of marimo's generic "ancestor stopped".
    def _probe_table(probe):
        t = probe.get("torch")
        torch_txt = (f"available={t.get('available')} devices={t.get('count')} name={t.get('name')!r} "
                     f"capability={t.get('capability')} torch={t.get('torch')} cuda={t.get('cuda')}"
                     if isinstance(t, dict) else str(t)[-400:])
        return mo.ui.table([
            {"probe": "detected GPU", "result": f"{probe.get('name') or '(none)'}"
                                                f"{' via ' + probe['source'] if probe.get('source') else ''}"},
            {"probe": "nvidia-smi", "result": str(probe.get("nvidia_smi"))[-400:]},
            {"probe": "torch.cuda", "result": torch_txt},
            {"probe": "/proc/driver/nvidia", "result": str(probe.get("procfs"))},
            {"probe": "/dev/nvidia*", "result": ", ".join(probe.get("device_nodes") or []) or "none"},
            {"probe": "CUDA_VISIBLE_DEVICES", "result": str(probe.get("cuda_visible_devices"))},
        ], selection=None, pagination=False)

    def _connect(v):
        """Returns (report sections, grader, queue, worker_id); grader/queue are None unless fully ready."""
        wid = new_worker_id("molab")
        out = [mo.md("### Connection report")]

        # 1) W&B read access (identity, project queries)
        try:
            q = WandbQueue(v["entity"].strip(), v["project"].strip(), api_key=v["key"].strip())
            viewer = q.api.viewer
            user = getattr(viewer, "username", None) or getattr(viewer, "name", "?")
            hb = q.worker_heartbeat_age()
            out.append(mo.callout(mo.md(
                f"**W&B read:** signed in as **{user}**, project `{q.entity}/{q.project}`  \n"
                f"previous worker heartbeat: {'none' if hb is None else f'{hb / 60:.1f} min ago'}"), kind="success"))
        except Exception as e:
            out.append(mo.callout(mo.md(f"**W&B read:** failed. {explain_wandb_error(e, v['entity'])}"), kind="danger"))
            return out, None, None, None

        # 2) W&B write access (the worker creates a run and updates candidate runs)
        try:
            q.check_write_access(wid)
            out.append(mo.callout(mo.md(f"**W&B write:** worker run `{wid}` created"), kind="success"))
        except Exception as e:
            out.append(mo.callout(mo.md(f"**W&B write:** failed. {explain_wandb_error(e, q.entity)}"), kind="danger"))
            return out, None, None, None

        # 3) deck
        if not deck_problems:
            out.append(mo.callout(mo.md("**Deck:** not available, see section 2."), kind="danger"))
            return out, None, None, None
        out.append(mo.callout(mo.md(f"**Deck:** `{cfg.deck.commit[:10]}`, {len(deck_problems)} problems"),
                              kind="success"))

        # 4) GPU + grader
        def make_grader(expected):
            return KernelBenchGrader(cfg.path(cfg.deck.local_path), cfg.deck.subdir, cfg.deck.problems_dir,
                                     cfg.deck.commit, expected, cfg.worker.check_timeout_seconds,
                                     cfg.worker.bench_timeout_seconds, worker_id=wid,
                                     toolkit_root=cfg.path(".looppp"))
        try:
            g = make_grader(v["expected_gpu"].strip() or None)
            out.append(mo.callout(mo.md(f"**GPU:** `{g.gpu}` (via {g.probe.get('source')})"), kind="success"))
        except GpuCheckError as e:
            if not v["allow_unverified_gpu"]:
                t = e.probe.get("torch")
                hint = ("PyTorch cannot see a GPU: attach the RTX PRO 6000 with the notebook specs button, then "
                        "restart the kernel and run all cells." if not (isinstance(t, dict) and t.get("available"))
                        else "A GPU is visible but its name does not match. If the detected name is right, change "
                             "'expected GPU' in the form (or tick the override) and press Connect again.")
                out += [mo.callout(mo.md(f"**GPU:** {e}  \n{hint}"), kind="danger"), _probe_table(e.probe)]
                return out, None, None, None
            g = make_grader(None)
            out.append(mo.callout(mo.md(f"**GPU:** check overridden ({e}). Grades will record gpu_name={g.gpu!r}."),
                                  kind="warn"))
        except Exception as e:
            out += [mo.callout(mo.md(f"**Grader:** not ready: `{type(e).__name__}: {e}`"), kind="danger"),
                    _probe_table(gpu_probe(sys.executable))]
            return out, None, None, None

        home = g.cuda_home()
        out += [
            mo.callout(mo.md(f"**CUDA toolkit:** `{home}`" if home else
                             "**CUDA toolkit:** none. C++/CUDA extension solutions will fail with `toolchain`; "
                             "see section 2b."), kind="success" if home else "warn"),
            _probe_table(g.probe),
            mo.callout(mo.md(f"**Grader ready.** torch `{g.torch}` · worker id `{wid}`  \n"
                             "Next: calibrate (section 4), then start the worker (section 5)."), kind="success"),
        ]
        return out, g, q, wid

    _v = connect_form.value
    if _v is None:
        _report, grader, queue, worker_id = [mo.md("Fill in the form and press **Connect**. After a kernel "
                                                    "restart the form is empty again: re-enter the key.")], None, None, None
    elif not _v["key"].strip() or not _v["entity"].strip():
        _report, grader, queue, worker_id = [mo.callout(mo.md("Key and entity are required."), kind="warn")], None, None, None
    else:
        with mo.status.spinner(title="Connecting: W&B, deck, GPU..."):
            _report, grader, queue, worker_id = _connect(_v)
    mo.vstack(_report)
    return grader, queue, worker_id


@app.cell
def _(cfg, deck_problems, mo):
    _options = {
        f"{name} ({t.problem}, published {t.published_peak_fraction:.3f}"
        f"{', needs CUDA toolkit' if t.needs_cuda_toolkit else ', Triton'})": name
        for name, t in cfg.calibration.items() if t.problem in deck_problems
    }
    calib_form = mo.ui.dictionary({
        "target": mo.ui.dropdown(_options, value=next(iter(_options), None), label="published solution"),
        "runs": mo.ui.slider(1, 5, value=3, show_value=True, label="runs"),
    }).form(submit_button_label="Run calibration")
    mo.vstack([
        mo.md("## 4. Calibrate this GPU (recommended once per session)\nRebuilds a published KernelBench "
              "solution from its HuggingFace trace and grades it here. A ratio well below 1.0 means this GPU is "
              "slower than KernelBench's; looppp scores then compare with each other, not with the leaderboard. "
              "Targets marked *needs CUDA toolkit* require section 2b."),
        calib_form,
    ])
    return (calib_form,)


@app.cell
def _(calib_form, calibrate, calibration_solution, cfg, explain_wandb_error, grader, holder, mo,
      needs_cuda_toolkit, queue, worker_id):
    mo.stop(calib_form.value is None)
    mo.stop(grader is None or queue is None, mo.callout(mo.md("**Not connected.** Complete section 3 (Connect) first; its report shows what is missing."), kind="warn"))
    mo.stop(holder["worker"] is not None and holder["worker"].is_running,
            mo.callout(mo.md("Stop the worker first: one GPU job at a time keeps timings clean."), kind="warn"))
    _name = calib_form.value["target"]
    _t = cfg.calibration[_name]
    _p = _t.problem
    try:
        _code, _source = calibration_solution(_t.run_id, cfg.deck.repo, cfg.deck.commit, cfg.path(".looppp/traces"))
    except Exception as _e:
        mo.stop(True, mo.callout(mo.md(f"Cannot use `{_name}` for calibration: {_e}"), kind="danger"))
    mo.stop(needs_cuda_toolkit(_code) and not grader.cuda_home(),
            mo.callout(mo.md(f"`{_name}` builds a C++/CUDA extension and this worker has no CUDA toolkit. "
                             "Fix section 2b first, or pick a Triton target."), kind="warn"))
    with mo.status.spinner(title=f"Calibrating {_name}: {calib_form.value['runs']} gradings..."):
        calib_report = {"target": _name, "solution_source": _source, **calibrate(grader, _p, _code, calib_form.value["runs"],
                                                      _t.published_peak_fraction)}
    try:
        queue.worker_log(worker_id, {f"calibration/{_name}": calib_report})
        _logged = "Saved to the worker run in W&B."
    except Exception as _e:
        _logged = f"Not saved to W&B: {explain_wandb_error(_e, queue.entity)}"
    mo.callout(mo.md(f"```\n{calib_report}\n```\n{_logged}"),
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
    mo.stop(grader is None or queue is None, mo.callout(mo.md("**Not connected.** Complete section 3 (Connect) first; its report shows what is missing."), kind="warn"))
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
def _(explain_wandb_error, holder, mo, queue, refresh):
    refresh.value  # re-run on every tick
    mo.stop(queue is None, mo.callout(mo.md("**Not connected.** Complete section 3 (Connect) first; its report shows what is missing."), kind="warn"))
    _w = holder["worker"]
    _st = _w.status if _w is not None else {"state": "not started", "graded": 0, "errors": 0, "current": "",
                                           "last": None, "error": ""}
    _error = (mo.callout(mo.md(f"**Worker stopped with an error:** {explain_wandb_error(Exception(_st['error']), queue.entity)}"),
                         kind="danger") if _st.get("error") else mo.md(""))
    try:
        _rows = [{
            "status": r.status,
            "peak_fraction": r.result.peak_fraction if r.result else None,
            "fail_stage": r.result.fail_stage if r.result else None,
            "problem": r.spec.problem, "gen": f"{r.spec.generation}.{r.spec.index}",
            "model": r.spec.model, "hypothesis": r.spec.hypothesis[:90],
        } for r in queue.list_candidates(limit=15)]
    except Exception as _e:
        _rows = [{"error": explain_wandb_error(_e, queue.entity)}]
    _last = (_st.get("last") or {}).get("peak_fraction")
    mo.vstack([
        _error,
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
