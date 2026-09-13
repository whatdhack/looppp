import pytest

from looppp import contract as c
from looppp.queue.local import LocalQueue


def _spec(code: str, gen=0, idx=0, loop="L"):
    return c.CandidateSpec("01_toy", "commit", loop, gen, idx, "m", "h", c.sha256_text(code))


def test_submit_claim_complete(tmp_path):
    q = LocalQueue(tmp_path)
    cid = q.submit(_spec("a"), "a")
    assert q.get(cid).status == c.STATUS_PENDING
    rec, code = q.claim_next("w1")
    assert (rec.id, code, rec.status) == (cid, "a", c.STATUS_RUNNING)
    assert q.claim_next("w1") is None
    q.complete(cid, c.GradeResult(correct=True, peak_fraction=0.2))
    done = q.get(cid)
    assert done.status == c.STATUS_DONE and done.result.peak_fraction == 0.2 and done.terminal


def test_claim_order_and_sha_guard(tmp_path):
    q = LocalQueue(tmp_path)
    first = q.submit(_spec("a", idx=0), "a")
    q.submit(_spec("b", idx=1), "b")
    assert q.claim_next("w")[0].id == first
    with pytest.raises(ValueError):
        q.submit(_spec("a"), "not a")


def test_orphans_requeue_then_error(tmp_path):
    q = LocalQueue(tmp_path)
    cid = q.submit(_spec("a"), "a")
    q.claim_next("dead-worker")
    assert q.requeue_orphans(max_attempts=2) == [cid]
    rec = q.get(cid)
    assert rec.status == c.STATUS_PENDING and rec.attempts == 1
    q.claim_next("dead-worker-2")
    q.requeue_orphans(max_attempts=2)
    rec = q.get(cid)
    assert rec.status == c.STATUS_ERROR and rec.result.fail_stage == c.STAGE_WORKER_LOST


def test_abandon_and_heartbeat(tmp_path):
    q = LocalQueue(tmp_path)
    assert q.worker_heartbeat_age() is None
    q.heartbeat("w", {"graded": 0})
    assert q.worker_heartbeat_age() < 5
    cid = q.submit(_spec("a"), "a")
    q.abandon(cid, "timeout")
    assert q.get(cid).status == c.STATUS_ABANDONED
    assert q.claim_next("w") is None
    assert [r.id for r in q.list_candidates(loop_id="L")] == [cid]
    assert q.list_candidates(loop_id="other") == []
