"""Single-process mode: agent thread + grader thread + local queue."""
import time

import pytest

from looppp.grade import FakeGrader
from looppp.llm import StubLLM
from looppp.session import LoopSession, SessionOptions
from looppp.tracker import ConsoleTracker, TeeTracker
from tests.conftest import reply

MODEL = "class Model:\n    def forward(self, x):\n        return x * 2\n"


def _wait(pred, timeout=20.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.05)
    return False


def _session(cfg, toy_problem, tmp_path, llm, **opts):
    options = SessionOptions(poll_seconds=0.05, **opts)
    return LoopSession(cfg, toy_problem, llm, FakeGrader("fake"), "S1", "commit", options, state_root=tmp_path / "s")


def test_session_runs_to_completion(cfg, toy_problem, tmp_path):
    scores = [0.1, 0.3, 0.2, 0.25]
    llm = StubLLM(factory=lambda m, i: reply(f"idea {i}", f"{MODEL}# FAKE_SCORE={scores[i]}\n"))
    s = _session(cfg, toy_problem, tmp_path, llm, candidates_per_generation=2, max_generations=2)
    s.start()
    assert _wait(lambda: not s.is_running)
    snap = s.snapshot()
    assert snap["state"] == "finished" and snap["stop_reason"] == "max_generations"
    assert snap["graded"] == 4 and snap["best_peak_fraction"] == pytest.approx(0.3)
    assert snap["best_so_far"] == [0.1, 0.3, 0.3, 0.3]
    assert [r["gen"] for r in snap["rows"]] == ["0.0", "0.1", "1.0", "1.1"]
    code, meta = s.best()
    assert "FAKE_SCORE=0.3" in code and meta["label"] == "gen 0.1"
    assert any("[grader]" in line for line in snap["logs"]) and any("[agent]" in line for line in snap["logs"])
    assert s.archive.best_peak("01_toy") == pytest.approx(0.3)
    assert not s.worker.is_running
    assert snap["alerts"] == [], "no spurious worker-offline alert at startup"


def test_session_stop_is_prompt(cfg, toy_problem, tmp_path):
    llm = StubLLM(factory=lambda m, i: reply(f"idea {i}", f"{MODEL}# FAKE_SCORE=0.2\n# {i}\n"))
    s = _session(cfg, toy_problem, tmp_path, llm, candidates_per_generation=1, max_generations=1000, patience=1000)
    s.start()
    assert _wait(lambda: s.snapshot()["graded"] >= 2)
    s.stop()
    assert _wait(lambda: not s.is_running, timeout=10)
    snap = s.snapshot()
    assert snap["stop_reason"] == "stopped" and snap["state"] == "finished"
    assert snap["attempts"] < 1000


def test_session_surfaces_crash(cfg, toy_problem, tmp_path):
    class Boom:
        model = "boom"

        def complete(self, messages):
            raise RuntimeError("inference endpoint down")

    s = _session(cfg, toy_problem, tmp_path, Boom(), max_generations=1)
    s.start()
    assert _wait(lambda: not s.is_running)
    snap = s.snapshot()
    assert snap["state"] == "failed" and "inference endpoint down" in snap["error"]


def test_tee_tracker_isolates_secondary_failures(tmp_path):
    class Broken:
        def log(self, d): raise RuntimeError("wandb down")
        def summary(self, d): raise RuntimeError("wandb down")
        def alert(self, t, x): raise RuntimeError("wandb down")
        def finish(self): raise RuntimeError("wandb down")

    primary = ConsoleTracker(echo=False)
    t = TeeTracker(primary, Broken())
    t.log({"a": 1})
    t.alert("title", "text")
    t.finish()
    assert t.alerts == [("title", "text")]
