"""`looppp` command line."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

from looppp.config import Config, load_config


def load_dotenv(path: Path) -> None:
    """Minimal .env reader (KEY=VALUE lines); never overrides variables already set."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def _sanitize_id(text: str, limit: int = 64) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", text).strip("-")[:limit]


def _api_key(required: bool) -> str | None:
    key = os.environ.get("WANDB_API_KEY")
    if required and not key:
        sys.exit("WANDB_API_KEY is not set (put it in .env or export it)")
    return key


def _deck(cfg: Config):
    from looppp.problems import fetch_deck

    return fetch_deck(cfg.deck.repo, cfg.deck.commit, cfg.deck.subdir, cfg.path(cfg.deck.local_path))


# --- commands -------------------------------------------------------------------------
def cmd_fetch_deck(cfg: Config, args) -> None:
    from looppp.problems import list_problems

    _deck(cfg)
    print(f"deck {cfg.deck.commit[:10]} at {cfg.problems_root}")
    print("problems:", ", ".join(list_problems(cfg.problems_root)))


def cmd_run(cfg: Config, args) -> None:
    from looppp.agent import AgentLoop
    from looppp.archive import Archive
    from looppp.llm import StubLLM, WandbInferenceLLM
    from looppp.problems import load_problem
    from looppp.queue import make_queue
    from looppp.tracker import ConsoleTracker, WandbTracker

    ls = cfg.loop
    ls.problem = args.problem or ls.problem
    ls.model = args.model or ls.model
    if args.max_generations is not None:
        ls.max_generations = args.max_generations
    if args.candidates is not None:
        ls.candidates_per_generation = args.candidates
    if args.poll_seconds is not None:
        cfg.queue.poll_seconds = args.poll_seconds
    backend = args.backend or cfg.queue.backend

    _deck(cfg)
    problem = load_problem(cfg.problems_root, ls.problem)

    if args.stub_llm:
        ref = problem.reference

        def factory(messages, i):
            return f"HYPOTHESIS: stub variant {i}\n\n```python\n{ref}\n# variant {i}\n# FAKE_SCORE={0.05 + (i * 37 % 30) / 100:.2f}\n```"

        llm = StubLLM(factory=factory, model="stub")
    else:
        key = _api_key(required=True)
        project = f"{cfg.queue.entity}/{cfg.queue.project}" if cfg.queue.entity else None
        llm = WandbInferenceLLM(ls.model, cfg.model_settings(ls.model), api_key=key, project=project)
        if backend == "wandb" and not args.no_weave and cfg.queue.entity:
            import weave

            weave.init(f"{cfg.queue.entity}/{cfg.queue.project}")

    loop_id = args.loop_id or _sanitize_id(f"{ls.problem}-{llm.model.split('/')[-1]}-{time.strftime('%Y%m%d-%H%M%S')}")
    key = _api_key(required=backend == "wandb")
    queue = make_queue(cfg, api_key=key, backend=backend)
    loop_dir = cfg.path(ls.state_dir) / loop_id
    if backend == "wandb":
        tracker = WandbTracker(cfg.queue.entity, cfg.queue.project, loop_id,
                               {"problem": ls.problem, "model": ls.model, "deck_commit": cfg.deck.commit,
                                "loop": vars(ls)}, api_key=key)
    else:
        tracker = ConsoleTracker(loop_dir / "events.jsonl")
    if args.stub_llm:  # dry-run scores are fake: keep them out of the real archive
        archive = Archive(cfg.path(".looppp/archive-dryrun"))
    else:
        archive = Archive(cfg.path(cfg.archive.dir), git_commit=cfg.archive.git_commit and not args.no_archive_commit)

    print(f"loop {loop_id}: problem={ls.problem} model={llm.model} backend={backend} state={loop_dir}")
    loop = AgentLoop(cfg, problem, queue, llm, tracker, archive, loop_id, cfg.deck.commit)
    try:
        outcome = loop.run()
    finally:
        tracker.finish()
        queue.close()
    print(json.dumps({"loop_id": outcome.loop_id, "stop_reason": outcome.stop_reason,
                      "generations": outcome.generations, "attempts": outcome.attempts,
                      "best_peak_fraction": outcome.best.peak if outcome.best else None}, indent=2))


def cmd_worker(cfg: Config, args) -> None:
    from looppp.grade import FakeGrader, KernelBenchGrader
    from looppp.queue import make_queue
    from looppp.worker import Worker, new_worker_id

    backend = args.backend or cfg.queue.backend
    queue = make_queue(cfg, api_key=_api_key(required=backend == "wandb"), backend=backend)
    wid = new_worker_id("fake" if args.fake else "worker")
    if args.fake:
        grader = FakeGrader(wid, delay_seconds=args.fake_delay)
    else:
        _deck(cfg)
        _cuda_toolkit(cfg, install=args.install_cuda_toolkit)
        w = cfg.worker
        grader = KernelBenchGrader(cfg.path(cfg.deck.local_path), cfg.deck.subdir, cfg.deck.problems_dir,
                                   cfg.deck.commit, w.expected_gpu, w.check_timeout_seconds,
                                   w.bench_timeout_seconds, worker_id=wid, toolkit_root=cfg.path(".looppp"))
    worker = Worker(queue, grader, poll_seconds=args.poll_seconds or cfg.worker.poll_seconds,
                    heartbeat_seconds=cfg.worker.heartbeat_seconds, max_attempts=cfg.queue.max_attempts)
    print(f"worker {wid} ({backend}) grading with {type(grader).__name__}; Ctrl+C to stop")
    try:
        worker.run(max_candidates=args.max_candidates, idle_exit_seconds=args.idle_exit)
    except KeyboardInterrupt:
        worker.stop()
    finally:
        queue.close()
    print(json.dumps(worker.status, indent=2, default=str))


def cmd_status(cfg: Config, args) -> None:
    from looppp.queue import make_queue

    backend = args.backend or cfg.queue.backend
    queue = make_queue(cfg, api_key=_api_key(required=backend == "wandb"), backend=backend)
    age = queue.worker_heartbeat_age()
    print(f"worker heartbeat: {'never' if age is None else f'{age:.0f}s ago'}")
    for r in queue.list_candidates(loop_id=args.loop_id, limit=args.limit):
        g = r.result
        score = f"{g.peak_fraction:.4f}" if g and g.peak_fraction is not None else "-"
        stage = (g.fail_stage or "") if g else ""
        print(f"{r.status:10} {score:>7} {stage:12} g{r.spec.generation}.{r.spec.index} {r.spec.model:28} "
              f"{r.spec.hypothesis[:70]}")


def _cuda_toolkit(cfg: Config, install: bool) -> None:
    from looppp.cudatk import ensure_cuda_toolkit

    rep = ensure_cuda_toolkit(sys.executable, cfg.path(".looppp"), install=install)
    if rep.ok:
        print(f"CUDA toolkit: {rep.cuda_home} (nvcc {rep.nvcc_version}, via {rep.source})")
    else:
        print("CUDA toolkit: not available; CUDA C++ (load_inline) solutions will fail to build. "
              + ("" if install else "Pass --install-cuda-toolkit to install nvcc from pip. ")
              + " | ".join(rep.log)[-600:])


def cmd_cuda_toolkit(cfg: Config, args) -> None:
    from looppp.cudatk import ensure_cuda_toolkit

    rep = ensure_cuda_toolkit(sys.executable, cfg.path(".looppp"), install=not args.check_only)
    print(json.dumps(vars(rep), indent=2))
    if not rep.ok:
        sys.exit(1)


def cmd_calibrate(cfg: Config, args) -> None:
    from looppp.grade import KernelBenchGrader
    from looppp.traces import solution_from_run
    from looppp.worker import calibrate, new_worker_id

    target = cfg.calibration.get(args.target)
    if not target:
        sys.exit(f"no calibration target {args.target!r}; configured: {', '.join(cfg.calibration)}")
    _deck(cfg)
    _cuda_toolkit(cfg, install=args.install_cuda_toolkit)
    w = cfg.worker
    grader = KernelBenchGrader(cfg.path(cfg.deck.local_path), cfg.deck.subdir, cfg.deck.problems_dir,
                               cfg.deck.commit, w.expected_gpu, w.check_timeout_seconds, w.bench_timeout_seconds,
                               worker_id=new_worker_id("calibrate"), toolkit_root=cfg.path(".looppp"))
    code = solution_from_run(target.run_id, cfg.path(".looppp/traces"))
    report = calibrate(grader, target.problem, code, args.runs, target.published_peak_fraction)
    print(json.dumps({"target": args.target, **report}, indent=2))


def cmd_trace_solution(cfg: Config, args) -> None:
    from looppp.traces import solution_from_run

    code = solution_from_run(args.run_id, cfg.path(".looppp/traces"))
    if args.output:
        Path(args.output).write_text(code)
        print(f"wrote {len(code)} chars to {args.output}")
    else:
        sys.stdout.write(code)


def cmd_smoke_llm(cfg: Config, args) -> None:
    from looppp.llm import WandbInferenceLLM
    from looppp.parsing import ResponseFormatError, parse_llm_response

    model = args.model or cfg.loop.model
    project = f"{cfg.queue.entity}/{cfg.queue.project}" if cfg.queue.entity else None
    llm = WandbInferenceLLM(model, cfg.model_settings(model), api_key=_api_key(required=True), project=project)
    t0 = time.monotonic()
    reply = llm.complete([
        {"role": "system", "content": "Reply exactly in the requested format."},
        {"role": "user", "content": "HYPOTHESIS: <one line>\n\n```python\n<code>\n```\n\nWrite a PyTorch "
                                    "`class Model(torch.nn.Module)` whose forward returns x * 2."},
    ])
    try:
        hyp, code = parse_llm_response(reply.text)
        parsed = f"OK (hypothesis={hyp!r}, {len(code)} chars of code)"
    except ResponseFormatError as e:
        parsed = f"FORMAT ERROR: {e}"
    print(json.dumps({"model": model, "seconds": round(time.monotonic() - t0, 1), "finish_reason": reply.finish_reason,
                      "attempts": reply.attempts, "max_tokens_used": reply.max_tokens_used,
                      "prompt_tokens": reply.prompt_tokens, "completion_tokens": reply.completion_tokens,
                      "parse": parsed}, indent=2))


def cmd_smoke_queue(cfg: Config, args) -> None:
    """Round trip through the queue with a fake grader, in a separate project so no real
    candidate can be claimed by mistake."""
    from looppp import contract as c
    from looppp.grade import FakeGrader
    from looppp.queue import make_queue

    backend = args.backend or cfg.queue.backend
    cfg.queue.project = args.project
    queue = make_queue(cfg, api_key=_api_key(required=backend == "wandb"), backend=backend)
    code = "class Model:\n    pass\n# FAKE_SCORE=0.42\n"
    spec = c.CandidateSpec("smoke", "fake", f"smoke-{time.strftime('%Y%m%d-%H%M%S')}", 0, 0, "none",
                           "queue smoke test", c.sha256_text(code))
    steps = {}
    t = time.monotonic()
    cid = queue.submit(spec, code)
    steps["submit"] = round(time.monotonic() - t, 1)
    queue.heartbeat("smoke-worker", {"graded": 0})
    t = time.monotonic()
    claimed = None
    for _ in range(20):  # search results can lag a few seconds behind writes
        claimed = queue.claim_next("smoke-worker")
        if claimed:
            break
        time.sleep(3)
    steps["claim"] = round(time.monotonic() - t, 1)
    if not claimed or claimed[0].id != cid or claimed[1] != code:
        sys.exit(f"FAILED: claimed {claimed[0].id if claimed else None}, expected {cid}")
    queue.complete(cid, FakeGrader("smoke-worker").grade("smoke", code))
    rec = queue.get(cid)
    age = queue.worker_heartbeat_age()
    queue.close()
    ok = rec.status == c.STATUS_DONE and rec.result and rec.result.peak_fraction == 0.42
    print(json.dumps({"ok": bool(ok), "backend": backend, "project": args.project, "candidate": cid,
                      "status": rec.status, "peak_fraction": rec.result.peak_fraction if rec.result else None,
                      "heartbeat_age_s": None if age is None else round(age, 1), "seconds": steps}, indent=2))
    if not ok:
        sys.exit(1)


def cmd_env(cfg: Config, args) -> None:
    from looppp.envcheck import env_report

    for row in env_report(expected_gpu=cfg.worker.expected_gpu):
        flag = "ok " if row["ok"] else ("MISSING" if row["required"] else "warn")
        print(f"{flag:8} {row['check']:45} {row['detail'][:120]}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="looppp", description=__doc__)
    p.add_argument("--root", type=Path, default=None, help="repo root (default: auto-detect)")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("fetch-deck", help="clone the pinned KernelBench problem deck")

    r = sub.add_parser("run", help="run the agent loop")
    r.add_argument("--problem")
    r.add_argument("--model")
    r.add_argument("--backend", choices=["wandb", "local"])
    r.add_argument("--loop-id", help="resume an existing loop")
    r.add_argument("--max-generations", type=int)
    r.add_argument("--candidates", type=int, help="candidates per generation")
    r.add_argument("--poll-seconds", type=float)
    r.add_argument("--stub-llm", action="store_true", help="offline scripted model (dry runs)")
    r.add_argument("--no-weave", action="store_true")
    r.add_argument("--no-archive-commit", action="store_true")

    w = sub.add_parser("worker", help="run the evaluator loop (GPU box, or --fake on any machine)")
    w.add_argument("--backend", choices=["wandb", "local"])
    w.add_argument("--fake", action="store_true", help="CPU fake grader")
    w.add_argument("--fake-delay", type=float, default=1.0)
    w.add_argument("--poll-seconds", type=float)
    w.add_argument("--max-candidates", type=int)
    w.add_argument("--idle-exit", type=float, help="exit after this many idle seconds")
    w.add_argument("--install-cuda-toolkit", action="store_true", help="pip-install nvcc if no CUDA toolkit is found")

    s = sub.add_parser("status", help="worker heartbeat + recent candidates")
    s.add_argument("--backend", choices=["wandb", "local"])
    s.add_argument("--loop-id")
    s.add_argument("--limit", type=int, default=30)

    cal = sub.add_parser("calibrate", help="grade a published KernelBench solution on this GPU")
    cal.add_argument("--target", default="w4a16-triton-fable-5", help="name under calibration: in run.yaml")
    cal.add_argument("--runs", type=int, default=3)
    cal.add_argument("--install-cuda-toolkit", action="store_true")

    ct = sub.add_parser("cuda-toolkit", help="find or pip-install nvcc + headers and assemble CUDA_HOME")
    ct.add_argument("--check-only", action="store_true")

    t = sub.add_parser("trace-solution", help="rebuild solution.py from a KernelBench HF trace")
    t.add_argument("run_id")
    t.add_argument("-o", "--output")

    sl = sub.add_parser("smoke-llm", help="one tiny request to check a W&B Inference model")
    sl.add_argument("--model")

    sq = sub.add_parser("smoke-queue", help="submit/claim/complete round trip in a throwaway project")
    sq.add_argument("--backend", choices=["wandb", "local"])
    sq.add_argument("--project", default="looppp-smoke")

    sub.add_parser("env", help="GPU box environment report")

    args = p.parse_args(argv)
    cfg = load_config(args.root)
    load_dotenv(cfg.root / ".env")
    cfg = load_config(args.root)  # re-read so $WANDB_ENTITY / $LOOPPP_PROJECT from .env apply
    handler = {
        "fetch-deck": cmd_fetch_deck, "run": cmd_run, "worker": cmd_worker, "status": cmd_status,
        "calibrate": cmd_calibrate, "trace-solution": cmd_trace_solution, "smoke-llm": cmd_smoke_llm,
        "smoke-queue": cmd_smoke_queue, "env": cmd_env, "cuda-toolkit": cmd_cuda_toolkit,
    }[args.command]
    try:
        handler(cfg, args)
    except Exception as e:
        if type(e).__module__.startswith("wandb"):
            from looppp.queue.wandb_queue import explain_wandb_error

            sys.exit(f"W&B error: {explain_wandb_error(e, cfg.queue.entity or '')}".replace("**", ""))
        raise


if __name__ == "__main__":
    main()
