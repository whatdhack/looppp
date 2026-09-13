"""End-to-end loop tests: StubLLM -> AgentLoop -> LocalQueue -> Worker(FakeGrader) thread."""
import pytest

from looppp import contract as c
from looppp.agent import LOCAL_FAILED, AgentLoop
from looppp.archive import Archive
from looppp.grade import FakeGrader
from looppp.llm import StubLLM
from looppp.queue.local import LocalQueue
from looppp.tracker import ConsoleTracker
from looppp.worker import Worker
from tests.conftest import reply

MODEL = "class Model:\n    def forward(self, x):\n        return x * 2\n"


def _candidate(score, tag=""):
    return f"{MODEL}# FAKE_SCORE={score}\n# {tag}\n"


@pytest.fixture
def worker_env(cfg):
    queue = LocalQueue(cfg.path(cfg.queue.local_root))
    worker = Worker(queue, FakeGrader("fake-w"), poll_seconds=0.02, heartbeat_seconds=0.05)
    worker.start_thread()
    yield queue, worker
    worker.stop(join_timeout=5)


def _loop(cfg, problem, queue, llm, loop_id="L1"):
    tracker = ConsoleTracker(echo=False)
    archive = Archive(cfg.path(cfg.archive.dir))
    return AgentLoop(cfg, problem, queue, llm, tracker, archive, loop_id, "commit", log=lambda s: None), tracker, archive


def test_full_loop_improves_and_archives(cfg, toy_problem, worker_env):
    queue, _ = worker_env
    scores = [0.10, 0.20, 0.15, 0.30, 0.25, 0.28]
    llm = StubLLM(factory=lambda msgs, i: reply(f"idea {i}", _candidate(scores[i], f"call {i}")))
    loop, tracker, archive = _loop(cfg, toy_problem, queue, llm)
    out = loop.run()

    assert out.stop_reason == "max_generations"
    assert len(loop.attempts) == 6
    assert all(a.status == c.STATUS_DONE for a in loop.attempts)
    assert out.best.peak == pytest.approx(0.30) and out.best.hypothesis == "idea 3"
    code, meta = archive.best("01_toy")
    assert "FAKE_SCORE=0.3" in code and meta["peak_fraction"] == pytest.approx(0.30)
    # the second generation's prompt shows the first generation's results
    gen1_prompt = llm.calls[2][1]["content"]
    assert "idea 0" in gen1_prompt and "peak_fraction=0.1000" in gen1_prompt
    # siblings are asked to try something different
    assert "idea 2" in llm.calls[3][1]["content"]


def test_precheck_repair_and_duplicates(cfg, toy_problem, worker_env):
    queue, _ = worker_env
    cfg.loop.max_generations = 1
    replies = [
        "HYPOTHESIS: forgot the class\n```python\ndef f():\n    return 1\n```",   # precheck fail -> repair
        reply("fixed", _candidate(0.2, "same")),
        reply("again", _candidate(0.2, "same")),                                   # identical code -> duplicate
    ]
    llm = StubLLM(replies=replies)
    loop, _, _ = _loop(cfg, toy_problem, queue, llm)
    loop.run()
    first, second = loop.attempts
    assert first.status == c.STATUS_DONE and first.hypothesis == "fixed"
    assert "pre-checks" in llm.calls[1][-1]["content"]
    assert second.status == LOCAL_FAILED and second.grade.fail_stage == c.STAGE_DUPLICATE
    assert len(queue.list_candidates()) == 1


def test_failure_logs_fed_back(cfg, toy_problem, worker_env):
    queue, _ = worker_env
    cfg.loop.max_generations = 2
    cfg.loop.candidates_per_generation = 1
    llm = StubLLM(replies=[reply("broken", MODEL + "# FAKE_FAIL\n"), reply("works", _candidate(0.2))])
    loop, _, _ = _loop(cfg, toy_problem, queue, llm)
    loop.run()
    assert loop.attempts[0].grade.fail_stage == c.STAGE_CHECK
    assert "Evaluator log for failed attempt gen 0.0" in llm.calls[1][1]["content"]
    assert loop.attempts[1].peak == pytest.approx(0.2)


def test_worker_down_pauses_then_stops(cfg, toy_problem):
    cfg.queue.max_worker_down_hours = 0.3 / 3600  # 0.3 s
    queue = LocalQueue(cfg.path(cfg.queue.local_root))  # no worker ever heartbeats
    llm = StubLLM(replies=[reply("x", _candidate(0.2))])
    loop, tracker, _ = _loop(cfg, toy_problem, queue, llm)
    out = loop.run()
    assert out.stop_reason == "worker_down"
    assert llm.calls == [], "no LLM spend while nobody can grade"
    assert tracker.alerts and "offline" in tracker.alerts[0][0]


def test_resume_waits_for_inflight_and_continues(cfg, toy_problem):
    cfg.loop.max_generations = 1
    cfg.loop.candidates_per_generation = 1
    queue = LocalQueue(cfg.path(cfg.queue.local_root))
    queue.heartbeat("pretend", {})
    llm = StubLLM(replies=[reply("first", _candidate(0.2))])
    loop, _, _ = _loop(cfg, toy_problem, queue, llm)
    loop._load()
    parent = loop._parent()
    loop.attempts.append(loop._propose_one(0, 0, 1, parent, []))  # submitted, then the agent "crashes"
    loop._save()

    worker = Worker(queue, FakeGrader("w"), poll_seconds=0.02, heartbeat_seconds=0.05)
    worker.start_thread()
    try:
        cfg.loop.max_generations = 2
        llm2 = StubLLM(replies=[reply("second", _candidate(0.3))])
        loop2, _, archive = _loop(cfg, toy_problem, queue, llm2)
        out = loop2.run()
    finally:
        worker.stop(join_timeout=5)
    assert [a.hypothesis for a in loop2.attempts] == ["first", "second"]
    assert all(a.status == c.STATUS_DONE for a in loop2.attempts)
    assert out.best.peak == pytest.approx(0.3)
    assert archive.best_peak("01_toy") == pytest.approx(0.3)
