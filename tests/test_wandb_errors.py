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
