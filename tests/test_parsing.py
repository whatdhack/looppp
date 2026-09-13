import pytest

from looppp import contract as c
from looppp.parsing import ResponseFormatError, parse_benchmark, parse_check, parse_llm_response

BENCH_OK = """\
shape=0 variant=solution tflops=1.2 gbps=300.1 ms=0.5
shape=0 solution_peak_fraction=0.2000
shape=1 variant=solution tflops=1.2 gbps=300.1 ms=0.5
shape=1 solution_peak_fraction=0.3000
peak_fraction: 0.2449
RESULT: OK
"""


def test_check_pass():
    assert parse_check(0, "compiling...\nPASS\n").passed


@pytest.mark.parametrize("log,stage", [
    ("FAIL: import error: No module named 'x'", c.STAGE_IMPORT),
    ("FAIL: forbidden op used: torch.nn.functional.linear", c.STAGE_FORBIDDEN),
    ("FAIL: shape 1 seed 42 case nominal: max abs err 3.1", c.STAGE_CHECK),
])
def test_check_fail_stages(log, stage):
    out = parse_check(1, log)
    assert not out.passed and out.fail_stage == stage


def test_check_fails_closed():
    assert parse_check(0, "all good I promise").fail_stage == c.STAGE_CHECK      # no PASS line
    assert parse_check(1, "PASS").fail_stage == c.STAGE_CHECK                     # PASS but non-zero exit
    assert parse_check(0, "PASS\nFAIL: late failure").fail_stage == c.STAGE_CHECK
    assert parse_check(0, "PASS", timed_out=True).fail_stage == c.STAGE_TIMEOUT


def test_benchmark_ok():
    out = parse_benchmark(0, BENCH_OK, expected_shapes=2)
    assert out.peak_fraction == pytest.approx(0.2449)
    assert out.shape_fractions == [0.2, 0.3]
    assert out.fail_stage is None


def test_benchmark_fails_closed():
    assert parse_benchmark(0, "shape=0 solution_peak_fraction=0.2").peak_fraction is None  # no final line
    assert parse_benchmark(1, BENCH_OK).fail_stage == c.STAGE_BENCHMARK
    assert parse_benchmark(0, BENCH_OK, expected_shapes=3).peak_fraction is None             # missing a shape
    faked = "peak_fraction: 0.99\n" + BENCH_OK                                               # candidate printed a marker
    assert parse_benchmark(0, faked, expected_shapes=2).peak_fraction is None
    assert parse_benchmark(0, BENCH_OK, timed_out=True).fail_stage == c.STAGE_TIMEOUT


def test_llm_response_picks_model_block():
    text = ("Some thinking.\nHYPOTHESIS: fuse unpack into the GEMM loop\n\n```python\nimport torch\n```\n"
            "and the file:\n```python\nimport torch\n\nclass Model(torch.nn.Module):\n    pass\n```\n")
    hyp, code = parse_llm_response(text)
    assert hyp == "fuse unpack into the GEMM loop"
    assert code.startswith("import torch\n\nclass Model")


def test_llm_response_markdown_hypothesis_and_missing_code():
    hyp, _ = parse_llm_response("**HYPOTHESIS:** tile M>1\n```py\nclass Model: pass\n```")
    assert hyp == "tile M>1"
    with pytest.raises(ResponseFormatError):
        parse_llm_response("HYPOTHESIS: nothing\nno code here")
