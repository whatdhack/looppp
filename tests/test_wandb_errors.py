from looppp.grade import FakeGrader
from looppp.queue.local import LocalQueue
from looppp.queue.wandb_queue import explain_wandb_error
from looppp.worker import Worker


def test_models_seat_message():
    msg = explain_wandb_error(Exception("user does not have models write access for this org"), "myteam")
    assert "Models seat" in msg and "Full" in msg and "`myteam`" in msg


def test_other_messages():
    assert "denied" in explain_wandb_error(Exception("permission denied"))
    assert "API key" in explain_wandb_error(Exception("401 Unauthorized"))
    assert explain_wandb_error(ValueError("boom")) == "ValueError: boom"


class _NoWriteQueue(LocalQueue):
    def worker_log(self, worker_id, data):
        raise RuntimeError("user does not have models write access for this org")


def test_worker_reports_startup_failure_instead_of_dying_silently(tmp_path):
    w = Worker(_NoWriteQueue(tmp_path), FakeGrader("w"), poll_seconds=0.01, heartbeat_seconds=0.01,
               log=lambda s: None)
    w.start_thread()
    w._thread.join(5)
    assert not w.is_running
    assert w.status["state"] == "failed"
    assert "models write access" in w.status["error"]


def test_calibrate_per_shape_diagnosis():
    from looppp.contract import GradeResult
    from looppp.worker import calibrate

    class G:
        worker_id = "g"
        def describe(self): return {}
        def grade(self, problem, code):
            return GradeResult(correct=True, peak_fraction=0.22, shape_fractions=[0.25, 0.30, 0.135, 0.15, 0.33])

    published = [{"idx": 0, "label": "a", "ms": 0.043, "frac": 0.345, "bound": "memory"},
                 {"idx": 1, "label": "b", "ms": 0.044, "frac": 0.348, "bound": "memory"},
                 {"idx": 2, "label": "c", "ms": 0.144, "frac": 0.135, "bound": "compute"},
                 {"idx": 3, "label": "d", "ms": 0.025, "frac": 0.197, "bound": "memory"},
                 {"idx": 4, "label": "e", "ms": 0.047, "frac": 0.377, "bound": "memory"}]
    rep = calibrate(G(), "p", "code", runs=2, published_peak_fraction=0.26, published_shapes=published)
    assert rep["shape_fractions"] == [0.25, 0.3, 0.135, 0.15, 0.33]
    assert rep["per_shape"][2]["ratio"] == 1.0 and rep["per_shape"][3]["ratio"] < 0.8
    assert "CPU" in rep["diagnosis"]
    uniform = [dict(p, frac=round(f / 0.86, 4)) for p, f in zip(published, [0.25, 0.30, 0.135, 0.15, 0.33])]
    assert "uniform" in calibrate(G(), "p", "c", 1, 0.26, uniform)["diagnosis"]
