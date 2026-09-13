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


def test_snapshot_reports_activity_while_model_is_thinking(cfg, toy_problem, tmp_path):
    import threading
    release = threading.Event()

    class SlowLLM:
        model = "slow"

        def complete(self, messages):
            release.wait(10)
            from looppp.llm import LLMReply
            return LLMReply(reply("slow idea", MODEL + "# FAKE_SCORE=0.2\n"), "stop")

    s = _session(cfg, toy_problem, tmp_path, SlowLLM(), candidates_per_generation=1, max_generations=1)
    s.start()
    assert _wait(lambda: "waiting for model" in s.snapshot()["activity"])
    snap = s.snapshot()
    assert snap["attempts"] == 0 and "gen 0.0 (candidate 1/1)" in snap["activity"]
    assert any("waiting for model (slow, prompt ~" in line for line in snap["logs"])
    release.set()
    assert _wait(lambda: not s.is_running)
    assert s.snapshot()["activity"] == ""


def test_seed_is_graded_first_used_as_parent_and_not_archived(cfg, toy_problem, tmp_path):
    seed_code = MODEL + "# FAKE_SCORE=0.25\n# published seed\n"
    llm = StubLLM(factory=lambda m, i: reply(f"idea {i}", f"{MODEL}# FAKE_SCORE=0.1\n# {i}\n"))
    options = SessionOptions(poll_seconds=0.05, candidates_per_generation=1, max_generations=1)
    s = LoopSession(cfg, toy_problem, llm, FakeGrader("fake"), "SEED", "commit", options, state_root=tmp_path / "s",
                    seed=(seed_code, "published fable-5"))
    s.start()
    assert _wait(lambda: not s.is_running)
    snap = s.snapshot()
    assert [r["gen"] for r in snap["rows"]] == ["seed", "0.0"]
    assert snap["rows"][0]["mode"] == "seed" and snap["rows"][0]["peak_fraction"] == pytest.approx(0.25)
    assert "published seed" in llm.calls[0][1]["content"], "model starts from the seed kernel"
    assert s.archive.best("01_toy") is None, "seed is not the loop's work"
    assert snap["best_peak_fraction"] == pytest.approx(0.25)


def test_explore_after_stagnation_and_timing(cfg, toy_problem, tmp_path):
    # constant score: gen 0 is the first significant result, gens 1-2 add nothing -> gen 3 explores
    llm = StubLLM(factory=lambda m, i: reply(f"tweak {i}", f"{MODEL}# FAKE_SCORE=0.2\n# {i}\n"))
    s = _session(cfg, toy_problem, tmp_path, llm, candidates_per_generation=2, max_generations=4, patience=10,
                 explore_after=2)
    s.start()
    assert _wait(lambda: not s.is_running, timeout=30)
    rows = s.snapshot()["rows"]
    modes = {r["gen"]: r["mode"] for r in rows}
    assert modes["0.0"] == modes["1.1"] == modes["2.1"] == "exploit"
    assert modes["3.0"] == "exploit" and modes["3.1"] == "explore"
    explore_prompt = llm.calls[7][1]["content"]
    assert "EXPLORE" in explore_prompt and "Approaches already tried" in explore_prompt and "tweak 0" in explore_prompt
    assert "EXPLORE" not in llm.calls[6][1]["content"], "one exploit candidate stays on the best kernel"
    snap = s.snapshot()
    assert snap["explore_attempts"] == 1
    assert snap["min_per_generation"] is not None and snap["generations_left_in_budget"] is not None
    assert all(r["total_min"] is not None for r in rows)


def test_top_k_parents_rotate(cfg, toy_problem, tmp_path):
    scores = [0.30, 0.20, 0.10, 0.10]
    llm = StubLLM(factory=lambda m, i: reply(f"idea {i}", f"{MODEL}# FAKE_SCORE={scores[i]}\n# kernel {i}\n"))
    s = _session(cfg, toy_problem, tmp_path, llm, candidates_per_generation=2, max_generations=2, parents_top_k=2)
    s.start()
    assert _wait(lambda: not s.is_running)
    gen1_prompts = [llm.calls[2][1]["content"], llm.calls[3][1]["content"]]
    assert "# kernel 0" in gen1_prompts[0].split("## Current parent")[1].split("## Best graded")[0]
    assert "# kernel 1" in gen1_prompts[1].split("## Current parent")[1].split("## Best graded")[0]
