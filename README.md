# looppp

An LLM agent loop that writes GPU kernels, graded on a real **RTX PRO 6000 Blackwell** in
[molab](https://molab.marimo.io), with [W&B](https://wandb.ai) carrying the work queue, results and
traces.

## Two ways to run

| | **Single notebook** (simplest) | **Distributed** |
|---|---|---|
| Notebook | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/whatdhack/looppp/blob/main/notebooks/loop.py) `notebooks/loop.py` | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/whatdhack/looppp/blob/main/worker/evaluator.py) `worker/evaluator.py` |
| Agent loop runs | in the molab notebook (thread) | on WSL (`looppp run`) |
| Grader runs | in the same notebook (thread) | in the molab notebook |
| Queue | local files on the molab disk | W&B runs |
| W&B needed for | the model API (W&B Inference); logging is optional | model API, queue, heartbeats (needs a Models seat) |
| Survives the tab/kernel dying | no: the loop stops with the session (resume by loop id) | yes: the WSL loop pauses and resumes |

### Single notebook

1. Open `notebooks/loop.py` in molab and attach the RTX PRO 6000.
2. Sections 1-2 check the environment, fetch the deck and install the CUDA toolkit automatically.
3. Section 3: paste the W&B API key, pick model and problem, press **Apply settings** (checks the GPU and
   builds the grader). **Test the model** sends one small request.
4. Section 4: **Start loop**. Section 5 shows best-so-far, the attempts table, the log, and the best kernel
   with a download button. **Stop loop** finishes the current step.

The rest of this README describes the distributed mode and the shared components.

```
WSL: looppp run (agent)                                   molab: worker/evaluator.py (RTX PRO 6000)
  1. prompt model (W&B Inference) with task, parent,        3. claim oldest pending candidate
     graded history, failure logs                           4. restore deck from git, drop in solution.py
  2. precheck locally, submit candidate  ──► W&B ◄──        5. check.py, benchmark.py in scrubbed subprocesses
  6. read results, pick best parent, archive improvements   6. write correct / peak_fraction / logs back
                                             │
                                             └──► ARIA / W&B UI: analyse candidates across loops
```

Problems come from the [KernelBench-Hard](https://github.com/Infatoshi/kernelbench.com) RTX PRO 6000
deck, pinned to one commit, so scores are comparable with its published leaderboard (after calibration).

## Quick start

### 1. WSL: install and check

```bash
cd ~/ai26/looppp
uv sync
cp .env.example .env            # fill WANDB_API_KEY and WANDB_ENTITY (a team)
uv run pytest -q                # offline: fake deck, fake worker, stub model
uv run looppp fetch-deck        # shallow sparse clone of the pinned deck into .looppp/deck
```

### 2. Dry run with no GPU and no LLM

Two terminals:

```bash
uv run looppp worker --backend local --fake              # terminal A: CPU fake grader
uv run looppp run --backend local --stub-llm --max-generations 2 --candidates 2 --poll-seconds 1
```

### 3. Check the real services

```bash
uv run looppp smoke-llm --model zai-org/GLM-5.2          # output format, tokens, finish_reason
uv run looppp smoke-queue --backend wandb                # W&B round trip in project looppp-smoke
```

Run `smoke-llm` for every model you plan to use. Reasoning models behave differently: some return
empty replies until `max_tokens` is raised (configs/models.yaml).

### 4. molab evaluator (manual start, at most 12 h per session)

1. Click **Open in molab** above and attach the **RTX PRO 6000** (notebook specs button).
2. **1. Environment**: every required row must be green.
3. **2b. CUDA toolkit**: press *Install CUDA compiler from pip* if it shows "No CUDA toolkit". molab's torch
   has the CUDA runtime but no `nvcc`, and CUDA C++ (`load_inline`) kernels need it. Triton kernels don't.
4. **3. Connect**: paste a W&B **service-account** key, enter the team entity, then press Connect.
5. **4. Calibrate** (once per session): grades a published KernelBench solution and compares it with
   its leaderboard score. `w4a16-triton-fable-5` and `paged-attention-triton-opus-4-8` work without the toolkit.
6. **5. Worker** → **Start worker**. Keep the tab open.

### 5. Real loop on WSL

```bash
uv run looppp run --problem 07_w4a16_gemm --model zai-org/GLM-5.2
uv run looppp status                          # worker heartbeat + recent candidates
uv run looppp run --loop-id <id>              # resume after a crash or restart
```

If the worker heartbeat goes stale, the loop **pauses** (no model spend), sends a W&B alert, and
resumes when a worker is back. After `queue.max_worker_down_hours` it stops with `worker_down`.

## Commands

| Command | Where | What |
|---|---|---|
| `looppp run` | WSL | Agent loop (`--stub-llm` for dry runs, `--loop-id` to resume) |
| `looppp worker` | GPU box / any (`--fake`) | Evaluator loop outside the notebook |
| `looppp status` | anywhere | Worker heartbeat age and recent candidates |
| `looppp fetch-deck` | anywhere | Pinned KernelBench deck |
| `looppp calibrate --target NAME` | GPU box | Grade a published solution (names under `calibration:` in run.yaml) |
| `looppp cuda-toolkit` | GPU box | Find, or pip-install, nvcc + headers and assemble `CUDA_HOME` |
| `looppp trace-solution RUN_ID -o f.py` | anywhere | Published graded solution.py of a run (`--replay` rebuilds it from the HF trace) |
| `looppp smoke-llm --model M` | WSL | One small W&B Inference request |
| `looppp smoke-queue` | anywhere | Queue round trip in a throwaway project |
| `looppp env` | GPU box | Environment report (same table as the notebook) |

## W&B layout (the contract, `src/looppp/contract.py`)

Every **candidate** is one W&B run: `job_type=candidate`, `group=<loop_id>`.

| Where | Fields |
|---|---|
| `config` (agent, immutable) | `kind`, `problem`, `deck_commit`, `loop_id`, `generation`, `index`, `model`, `hypothesis`, `code_sha256`, `parent` |
| file | `solution.py` |
| `summary` (worker) | `status` (`submitting → pending → running → done / error / abandoned`), `attempts`, `correct`, `peak_fraction`, `shape_fractions`, `fail_stage`, `fail_reason`, `check_tail`, `bench_tail`, `grade_seconds`, `gpu_name`, `torch_version`, `worker_id` |

A **worker** run (`job_type=worker`) logs `heartbeat_ts` and calibration results. An **agent** run
(`job_type=agent`, one per loop) logs per-generation best, failure counts and token usage, and sends
alerts. With Weave enabled (default for real runs), every W&B Inference call is traced.

`fail_stage` values: `llm`, `submit`, `precheck`, `duplicate` (agent side, never graded);
`download`, `integrity`, `import`, `forbidden`, `check`, `benchmark`, `timeout`, `tamper`,
`worker_lost`, `worker_exception` (worker side).

## Safety

Candidates are untrusted LLM-written code running next to your W&B key.

- **Key handling (molab):** password field only. The key is never written to disk or molab secrets
  and is passed explicitly to the W&B SDK (no `wandb.login`, no `~/.netrc`, no `os.environ`).
- **Grading subprocesses** get an environment with every secret-looking variable removed, run in
  their own process group with timeouts, and use per-candidate Triton and extension caches.
- **Tamper checks:** the deck subdir is restored from git before and after every candidate. Any change
  to tracked grader files fails the run (`tamper`).
- **Score integrity:** a score needs exactly one marker per shape (0..n-1) and exactly one
  `peak_fraction:` line. Extra markers printed by the candidate make the run unscored.
- **Key scope:** use a W&B **service account** key limited to the looppp team, so a leak is revocable
  and contained.

These close the easy paths. They are not a sandbox: in-process monkeypatching of timing code is
still possible. Treat suspiciously high scores the way KernelBench does, by reading the code before
trusting the number.

## Configuration

- `configs/run.yaml`: loop (problem, model, candidates per generation, stop rules), queue (backend,
  project, timeouts, heartbeat), deck (pinned commit), archive, worker timeouts, calibration targets.
- `configs/models.yaml`: W&B Inference model IDs with `max_tokens`, `max_tokens_cap`, `temperature`,
  `extra_body`.

Stop rules: `max_generations`, `patience` (generations without a new best), `wall_budget_hours`,
`target_peak_fraction`, and `worker_down`.

The best graded kernel per problem goes to `archive/<problem>/best.py` + `best.json`, only when it
beats the archived score for the same deck commit (`archive.git_commit: true` commits just those
files). Dry runs (`--stub-llm`) write to `.looppp/archive-dryrun/` instead.

## Layout

```
configs/            run.yaml, models.yaml
src/looppp/
  contract.py       shared schema (statuses, fail stages, CandidateSpec, GradeResult)
  config.py         typed config loading
  problems.py       pinned deck fetch + problem loading
  queue/            base protocol, local (files), wandb_queue
  agent.py          the loop: propose, precheck, submit, wait, select, archive, stop
  llm.py            W&B Inference client (empty/truncated reply retries), StubLLM
  prompts.py        system prompt, context and repair messages
  precheck.py       CPU-side syntax / interface / forbidden-op checks
  parsing.py        check.py / benchmark.py / model reply parsers (fail closed)
  grade.py          KernelBenchGrader (subprocess, scrubbed env, tamper checks), FakeGrader
  worker.py         claim-grade-report loop with heartbeat thread, calibration
  traces.py         rebuild published solutions from KernelBench HF traces
  tracker.py        W&B agent run / console tracker
  archive.py        best-per-problem archive
  envcheck.py       GPU box report, GPU probe (nvidia-smi / torch / procfs)
  cudatk.py         pip-provisioned CUDA toolkit (nvcc + headers) assembled into CUDA_HOME
  session.py        single-process mode: agent thread + grader thread + local queue
  cli.py            `looppp` commands
notebooks/loop.py   single notebook: loop + grader in molab
worker/evaluator.py distributed mode: grader-only notebook for molab
tests/              offline tests (fake deck git repo, fake worker, stub model)
```

## Known limits and things to verify on first real use

- **CUDA toolkit from pip:** only the compiler wheels (`nvcc`, `crt`, `nvvm`) are installed, at the newest
  release in torch's CUDA major version, with `--no-deps`. torch's pinned runtime libraries are never
  upgraded. The toolkit step compiles a small test kernel and upgrades the compiler once if an older minor
  fails against the host glibc (CUDA 13.0 fails on glibc 2.43).
- **molab behaviour** that isn't documented: whether a busy worker counts as "idle" for the 90-min
  shutdown, whether closing the tab kills the session, whether `nvcc` exists (Triton-only if not). The
  environment table and the heartbeat will show these.
- **One worker per W&B project.** Claiming is read-then-write, not atomic.
- **W&B search lag:** a freshly submitted candidate can take a few seconds to show up in queries.
- **Calibration** grades the kernel KernelBench published for that run
  (`public/runs/<run_id>_solution.py.txt` at the pinned deck commit), i.e. the file that produced the
  published score. Replaying the HF trace is only a fallback: it misses edits made through Bash and can
  land on an abandoned later version (it did for 3 of the 4 default targets).
- **`--root` / repo URL:** the notebook clones `https://github.com/whatdhack/looppp.git` when opened
  outside a checkout. Edit `LOOPPP_REPO` in `worker/evaluator.py` if the repo lives elsewhere.

Third-party material and licenses: [third_party/NOTICE.md](third_party/NOTICE.md).
