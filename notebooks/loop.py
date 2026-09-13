# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.24.2",
#     "openai>=3.13.0",
#     "wandb>=0.30.0",
#     "pyyaml>=6.0",
#     "hypothesis>=6.140",
#     "einops>=0.8",
#     "ninja>=1.11",
#     "numpy>=1.26",
# ]
# ///
"""looppp, all in one notebook: open in molab, attach the RTX PRO 6000, set up, start.

The agent loop (W&B Inference model) and the grader (this GPU) run as two threads in this
kernel and talk through a local queue on disk. No WSL process, no W&B queue. W&B is only
used for the model API and, optionally, for logging.
"""

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium", app_title="looppp loop")


@app.cell
def _():
    import importlib.util
    import os
    import subprocess
    import sys
    import time
    from pathlib import Path

    import marimo as mo

    return Path, importlib, mo, os, subprocess, sys, time


@app.cell
def _(Path, importlib, mo, subprocess, sys):
    LOOPPP_REPO = "https://github.com/whatdhack/looppp.git"  # edit if your repo lives elsewhere

    _here = Path(mo.notebook_dir() or ".").resolve()
    _is_repo = lambda p: (p / "src" / "looppp").is_dir() and (p / "configs" / "run.yaml").is_file()
    repo_root = next((p for p in (_here.parent, _here, Path.home() / "looppp") if _is_repo(p)), None)
    _log = []
    if repo_root is None:
        repo_root = Path.home() / "looppp"
        _r = subprocess.run(["git", "clone", "--depth", "1", LOOPPP_REPO, str(repo_root)], capture_output=True, text=True)
        _log.append(f"git clone: exit {_r.returncode} {_r.stderr.strip()[-300:]}")
    elif (repo_root / ".git").exists():
        _r = subprocess.run(["git", "-C", str(repo_root), "pull", "--ff-only", "-q"], capture_output=True, text=True)
        _log.append(f"git pull: exit {_r.returncode} {_r.stderr.strip()[-300:]}")
    if str(repo_root / "src") not in sys.path:
        sys.path.insert(0, str(repo_root / "src"))

    _needed = {"yaml": "pyyaml", "hypothesis": "hypothesis", "einops": "einops", "ninja": "ninja",
               "numpy": "numpy", "openai": "openai>=3.13.0", "wandb": "wandb>=0.30.0"}
    _missing = [pkg for mod, pkg in _needed.items() if importlib.util.find_spec(mod) is None]
    if _missing:
        _r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *_missing], capture_output=True, text=True)
        _log.append(f"pip install {' '.join(_missing)}: exit {_r.returncode} {_r.stderr.strip()[-300:]}")

    mo.vstack([
        mo.md(f"# looppp: agent loop + grader in one notebook\nCode: `{repo_root}`  \n"
              "Restarted the kernel? Re-run all cells, then re-enter the key in section 3."),
        mo.md("```\n" + ("\n".join(_log) or "code and dependencies already present") + "\n```"),
    ])
    return (repo_root,)


@app.cell
def _(repo_root):
    from looppp.cli import _sanitize_id as sanitize_id
    from looppp.config import load_config
    from looppp.cudatk import ensure_cuda_toolkit
    from looppp.envcheck import env_report, missing_required
    from looppp.grade import GpuCheckError, KernelBenchGrader
    from looppp.llm import WandbInferenceLLM
    from looppp.parsing import ResponseFormatError, parse_llm_response
    from looppp.problems import fetch_deck, list_problems, load_problem
    from looppp.queue.wandb_queue import explain_wandb_error
    from looppp.session import LoopSession, SessionOptions, sparkline_svg
    from looppp.tracker import ConsoleTracker, TeeTracker, WandbTracker

    cfg = load_config(repo_root)
    holder = {"session": None}  # survives re-runs of the cells below
    return (
        ConsoleTracker,
        GpuCheckError,
        KernelBenchGrader,
        LoopSession,
        ResponseFormatError,
        SessionOptions,
        TeeTracker,
        WandbInferenceLLM,
        WandbTracker,
        cfg,
        ensure_cuda_toolkit,
        env_report,
        explain_wandb_error,
        fetch_deck,
        holder,
        list_problems,
        load_problem,
        missing_required,
        parse_llm_response,
        sanitize_id,
        sparkline_svg,
    )


@app.cell
def _(cfg, env_report, fetch_deck, list_problems, missing_required, mo, sys):
    env_rows = env_report(python=sys.executable, expected_gpu=cfg.worker.expected_gpu)
    _missing = missing_required(env_rows)
    try:
        fetch_deck(cfg.deck.repo, cfg.deck.commit, cfg.deck.subdir, cfg.path(cfg.deck.local_path))
        deck_problems = list_problems(cfg.problems_root)
        _deck = mo.md(f"Pinned KernelBench deck `{cfg.deck.commit[:10]}`: " + ", ".join(f"`{p}`" for p in deck_problems))
    except Exception as _e:
        deck_problems = []
        _deck = mo.callout(mo.md(f"Deck fetch failed: `{type(_e).__name__}: {_e}`"), kind="danger")
    mo.vstack([
        mo.md("## 1. Environment and problems"),
        mo.callout(mo.md("All required checks passed." if not _missing else
                         f"**Missing:** {', '.join(_missing)}. Attach the GPU (notebook specs button), restart, re-run."),
                   kind="success" if not _missing else "danger"),
        _deck,
        mo.accordion({"Environment details": mo.ui.table(env_rows, selection=None, pagination=False)}),
    ])
    return (deck_problems,)


@app.cell
def _(mo):
    cuda_btn = mo.ui.run_button(label="Re-check / retry CUDA toolkit install")
    cuda_btn
    return (cuda_btn,)


@app.cell
def _(cfg, cuda_btn, ensure_cuda_toolkit, mo, sys):
    cuda_btn.value
    with mo.status.spinner(title="Checking / installing CUDA toolkit (nvcc + headers)..."):
        cuda_report = ensure_cuda_toolkit(sys.executable, cfg.path(".looppp"), install=True)
    mo.vstack([
        mo.md("## 2. CUDA toolkit"),
        mo.callout(mo.md(f"**Ready:** nvcc {cuda_report.nvcc_version} at `{cuda_report.cuda_home}` (torch CUDA "
                         f"{cuda_report.torch_cuda}). Triton and C++/CUDA extension kernels can be graded.")
                   if cuda_report.ok else
                   mo.md("**Install failed.** Triton kernels still grade; C++/CUDA extension kernels will fail with "
                         "`toolchain`.\n\n```\n" + "\n".join(cuda_report.log)[-1500:] + "\n```"),
                   kind="success" if cuda_report.ok else "danger"),
    ])
    return


@app.cell
def _(cfg, deck_problems, mo):
    _models = list(cfg.models) or [cfg.loop.model]
    settings_form = mo.ui.dictionary({
        "key": mo.ui.text(kind="password", label="WANDB_API_KEY (W&B Inference; memory only)", full_width=True),
        "model": mo.ui.dropdown(_models, value=cfg.loop.model if cfg.loop.model in _models else _models[0],
                                label="model (W&B Inference)"),
        "custom_model": mo.ui.text(value="", label="or another W&B Inference model id"),
        "problem": mo.ui.dropdown(deck_problems or [cfg.loop.problem],
                                  value=cfg.loop.problem if cfg.loop.problem in deck_problems else
                                  (deck_problems[0] if deck_problems else cfg.loop.problem), label="problem"),
        "candidates": mo.ui.slider(1, 6, value=cfg.loop.candidates_per_generation, show_value=True,
                                   label="candidates per generation"),
        "generations": mo.ui.number(1, 500, value=cfg.loop.max_generations, label="max generations"),
        "patience": mo.ui.number(1, 100, value=cfg.loop.patience, label="stop after N generations without a new best"),
        "wall_hours": mo.ui.number(0.1, 11.5, step=0.1, value=11.0, label="wall-clock budget (h; molab stops at 12)"),
        "target": mo.ui.number(0, 1, step=0.01, value=0, label="stop at peak_fraction (0 = off)"),
        "resume_loop_id": mo.ui.text(value="", label="resume loop id (optional)"),
        "expected_gpu": mo.ui.text(value=cfg.worker.expected_gpu or "", label="expected GPU (substring)"),
        "log_to_wandb": mo.ui.checkbox(value=False, label="also log generations to a W&B run (needs a Models seat)"),
        "weave": mo.ui.checkbox(value=False, label="trace model calls with Weave"),
        "entity": mo.ui.text(value=cfg.queue.entity or "", label="W&B entity (for logging / usage attribution)"),
        "project": mo.ui.text(value=cfg.queue.project, label="W&B project"),
    }).form(submit_button_label="Apply settings")
    mo.vstack([mo.md("## 3. Settings\nApplying settings checks the GPU and builds the grader. Nothing runs yet."),
               settings_form])
    return (settings_form,)


@app.cell
def _(GpuCheckError, KernelBenchGrader, WandbInferenceLLM, cfg, load_problem, mo, settings_form):
    # Never mo.stop here: export `ready = None` with a reason so later cells can say what is missing.
    def _prepare(v):
        out = []
        if not v["key"].strip():
            return None, [mo.callout(mo.md("Enter the W&B API key (used for W&B Inference)."), kind="warn")]
        model = v["custom_model"].strip() or v["model"]
        try:
            problem = load_problem(cfg.problems_root, v["problem"])
        except Exception as e:
            return None, [mo.callout(mo.md(f"**Problem:** `{e}`"), kind="danger")]
        project = f"{v['entity'].strip()}/{v['project'].strip()}" if v["entity"].strip() else None
        llm = WandbInferenceLLM(model, cfg.model_settings(model), api_key=v["key"].strip(), project=project)
        out.append(mo.callout(mo.md(f"**Model:** `{model}` · **Problem:** `{problem.name}` "
                                    f"(forbidden: {', '.join(problem.forbidden) or 'none'})"), kind="success"))
        try:
            grader = KernelBenchGrader(cfg.path(cfg.deck.local_path), cfg.deck.subdir, cfg.deck.problems_dir,
                                       cfg.deck.commit, v["expected_gpu"].strip() or None,
                                       cfg.worker.check_timeout_seconds, cfg.worker.bench_timeout_seconds,
                                       worker_id="notebook-grader", toolkit_root=cfg.path(".looppp"))
        except GpuCheckError as e:
            return None, out + [mo.callout(mo.md(f"**GPU:** {e}. Attach the RTX PRO 6000, restart the kernel, "
                                                 "re-run all cells."), kind="danger")]
        except Exception as e:
            return None, out + [mo.callout(mo.md(f"**Grader:** `{type(e).__name__}: {e}`"), kind="danger")]
        home = grader.cuda_home()
        out.append(mo.callout(mo.md(f"**GPU:** `{grader.gpu}` · torch `{grader.torch}` · CUDA toolkit: "
                                    f"{'`' + home + '`' if home else 'none (C++/CUDA kernels will fail)'}"),
                              kind="success" if home else "warn"))
        return {"settings": v, "model": model, "problem": problem, "llm": llm, "grader": grader}, out

    if settings_form.value is None:
        ready, _report = None, [mo.md("Fill in the settings and press **Apply settings**.")]
    else:
        with mo.status.spinner(title="Checking GPU and building the grader..."):
            ready, _report = _prepare(settings_form.value)
    mo.vstack(_report)
    return (ready,)


@app.cell
def _(mo):
    smoke_btn = mo.ui.run_button(label="Test the model (one small request)")
    smoke_btn
    return (smoke_btn,)


@app.cell
def _(ResponseFormatError, explain_wandb_error, mo, parse_llm_response, ready, smoke_btn, time):
    mo.stop(not smoke_btn.value)
    mo.stop(ready is None, mo.callout(mo.md("Apply valid settings first (section 3)."), kind="warn"))
    _t0 = time.monotonic()
    try:
        with mo.status.spinner(title=f"Calling {ready['model']}..."):
            _reply = ready["llm"].complete([
                {"role": "system", "content": "Reply exactly in the requested format."},
                {"role": "user", "content": "HYPOTHESIS: <one line>\n\n```python\n<code>\n```\n\nWrite a PyTorch "
                                            "`class Model(torch.nn.Module)` whose forward returns x * 2."},
            ])
        try:
            _hyp, _code = parse_llm_response(_reply.text)
            _parsed = f"format OK, hypothesis: {_hyp!r}"
        except ResponseFormatError as _e:
            _parsed = f"FORMAT PROBLEM: {_e}"
        _out = mo.callout(mo.md(
            f"**{ready['model']}** answered in {time.monotonic() - _t0:.1f}s · finish_reason `{_reply.finish_reason}` · "
            f"{_reply.completion_tokens} completion tokens · attempts {_reply.attempts}  \n{_parsed}"),
            kind="success" if _parsed.startswith("format OK") else "warn")
    except Exception as _e:
        _out = mo.callout(mo.md(f"**Model call failed:** {explain_wandb_error(_e)}"), kind="danger")
    _out
    return


@app.cell
def _(mo):
    start_btn = mo.ui.run_button(label="Start loop", kind="success")
    stop_btn = mo.ui.run_button(label="Stop loop", kind="danger")
    mo.vstack([mo.md("## 4. Run\nKeep this tab open. Stop takes effect after the current model call or grading."),
               mo.hstack([start_btn, stop_btn], justify="start")])
    return start_btn, stop_btn


@app.cell
def _(ConsoleTracker, LoopSession, SessionOptions, TeeTracker, WandbInferenceLLM, WandbTracker, cfg,
      explain_wandb_error, holder, mo, os, ready, sanitize_id, start_btn, time):
    mo.stop(not start_btn.value)
    mo.stop(ready is None, mo.callout(mo.md("Apply valid settings first (section 3)."), kind="warn"))
    _current = holder["session"]
    mo.stop(_current is not None and _current.is_running,
            mo.callout(mo.md(f"Loop `{_current.loop_id if _current else ''}` is already running."), kind="info"))

    _v, _notes = ready["settings"], []
    _loop_id = _v["resume_loop_id"].strip() or sanitize_id(
        f"{ready['problem'].name}-{ready['model'].split('/')[-1]}-{time.strftime('%Y%m%d-%H%M%S')}")
    _root = cfg.path(".looppp/session")
    _tracker = ConsoleTracker(_root / "loops" / _loop_id / "events.jsonl", echo=False)
    _entity, _project, _key = _v["entity"].strip(), _v["project"].strip(), _v["key"].strip()

    if _v["weave"]:
        try:
            import weave

            os.environ["WANDB_API_KEY"] = _key  # weave reads it; grading subprocesses never see it (scrubbed env)
            weave.init(f"{_entity}/{_project}" if _entity else _project)
            _notes.append("Weave tracing on.")
        except Exception as _e:
            _notes.append(f"Weave not started: {explain_wandb_error(_e, _entity)}")
    _llm = WandbInferenceLLM(ready["model"], cfg.model_settings(ready["model"]), api_key=_key,
                             project=f"{_entity}/{_project}" if _entity else None)  # created after weave.init
    if _v["log_to_wandb"]:
        try:
            _wb = WandbTracker(_entity, _project, _loop_id, {"problem": ready["problem"].name, "model": ready["model"],
                                                              "deck_commit": cfg.deck.commit, "mode": "notebook"},
                               api_key=_key)
            _tracker = TeeTracker(_tracker, _wb)
            _notes.append(f"Logging generations to W&B run `{_loop_id}`.")
        except Exception as _e:
            _notes.append(f"W&B logging off: {explain_wandb_error(_e, _entity)}")

    _target = float(_v["target"] or 0)
    _options = SessionOptions(candidates_per_generation=int(_v["candidates"]), max_generations=int(_v["generations"]),
                              patience=int(_v["patience"]), wall_budget_hours=float(_v["wall_hours"]),
                              target_peak_fraction=_target if _target > 0 else None, poll_seconds=5.0)
    holder["session"] = LoopSession(cfg, ready["problem"], _llm, ready["grader"], _loop_id, cfg.deck.commit,
                                    _options, tracker=_tracker, state_root=_root)
    holder["session"].start()
    mo.callout(mo.md(f"Started loop `{_loop_id}`. " + " ".join(_notes)), kind="success")
    return


@app.cell
def _(holder, mo, stop_btn):
    mo.stop(not stop_btn.value)
    if holder["session"] is not None and holder["session"].is_running:
        holder["session"].stop()
        _msg = "Stop requested; the loop finishes its current step."
    else:
        _msg = "No loop is running."
    mo.md(_msg)
    return


@app.cell
def _(mo):
    refresh = mo.ui.refresh(options=["5s", "10s", "30s"], default_interval="10s", label="auto-refresh")
    refresh
    return (refresh,)


@app.cell
def _(holder, mo, refresh, sparkline_svg):
    refresh.value  # re-run on every tick
    _s = holder["session"]
    mo.stop(_s is None, mo.md("## 5. Progress\nNo loop started yet."))
    _snap = _s.snapshot()
    _w = _snap["worker"]
    _best = _s.best()
    _state_kind = {"running": "info", "finished": "success", "failed": "danger", "stopping": "warn"}.get(_snap["state"], "neutral")
    _headline = (f"**{_snap['state']}**" + (f" ({_snap['stop_reason']})" if _snap["stop_reason"] else "")
                 + f" · loop `{_snap['loop_id']}` · {_snap['elapsed_min']} min"
                 + (f"  \n**Error:** {_snap['error']}" if _snap["error"] else ""))
    _parts = [
        mo.md("## 5. Progress"),
        mo.callout(mo.md(_headline), kind=_state_kind),
        mo.hstack([
            mo.stat(f"{_snap['best_peak_fraction']:.4f}" if _snap["best_peak_fraction"] is not None else "-",
                    label="best peak_fraction", caption=_snap["best_label"]),
            mo.stat(_snap["generation"] + 1, label="generation"),
            mo.stat(f"{_snap['graded']} / {_snap['attempts']}", label="graded / attempts"),
            mo.stat(str(_w.get("state", "-")), label="grader", caption=(_w.get("current") or "")[-24:]),
        ], widths="equal"),
        mo.Html(sparkline_svg(_snap["best_so_far"])),
    ]
    if _snap["alerts"]:
        _parts.append(mo.callout(mo.md("\n".join(f"- **{t}**: {x}" for t, x in _snap["alerts"][-3:])), kind="warn"))
    if _snap["rows"]:
        _parts.append(mo.ui.table(list(reversed(_snap["rows"])), selection=None, pagination=True, page_size=12,
                                  label="attempts (newest first)"))
    _acc = {"Log": mo.md("```\n" + "\n".join(_snap["logs"][-60:]) + "\n```")}
    if _best:
        _code, _meta = _best
        _acc[f"Best kernel ({_meta['label']}, {_meta['peak_fraction']:.4f})"] = mo.vstack([
            mo.md(f"**Hypothesis:** {_meta['hypothesis']}  \nper shape: {_meta['shape_fractions']}"),
            mo.download(data=_code.encode(), filename=f"{_s.problem.name}_best.py",
                        mimetype="text/x-python", label="Download best kernel"),
            mo.md(f"```python\n{_code}\n```"),
        ])
    _parts.append(mo.accordion(_acc))
    mo.vstack(_parts)
    return


if __name__ == "__main__":
    app.run()
