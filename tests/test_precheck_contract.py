from looppp import contract as c
from looppp.precheck import precheck
from looppp.problems import count_shapes


def test_precheck_ok():
    assert precheck("import torch\nclass Model(torch.nn.Module):\n    pass\n").ok


def test_precheck_errors():
    assert not precheck("").ok
    assert "SyntaxError" in precheck("class Model(:\n").errors[0]
    assert any("Model" in e for e in precheck("def f(): pass\n").errors)
    res = precheck("import torch\nclass Model: pass\ny = torch.nn.functional.linear\n", ["torch.nn.functional.linear"])
    assert any("forbidden" in e for e in res.errors)
    assert any("sys.exit" in e for e in precheck("import sys\nclass Model: pass\nsys.exit(0)\n").errors)


def test_precheck_accepts_imported_model():
    assert precheck("from kernels import Model\n").ok


def test_count_shapes():
    assert count_shapes("SHAPES = [{'M': 1}, {'M': 2}]\n") == 2
    assert count_shapes("SHAPES = make()\n") is None


def test_spec_and_result_roundtrip():
    spec = c.CandidateSpec("p", "abc", "loop", 1, 2, "m", "h", c.sha256_text("x"), parent=None)
    cfg = spec.to_config()
    assert cfg["kind"] == c.KIND_CANDIDATE
    assert c.CandidateSpec.from_config({**cfg, "_wandb": {}}) == spec
    r = c.GradeResult(correct=True, peak_fraction=0.3, shape_fractions=[0.3])
    back = c.GradeResult.from_summary({**r.to_summary(), "status": "done", "_runtime": 3})
    assert back == r and back.scored
    assert not c.GradeResult(correct=True).scored
