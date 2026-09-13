import subprocess
import time

import pytest

from looppp import contract as c
from looppp.grade import KernelBenchGrader, scrubbed_env
from tests.conftest import GOOD_SOLUTION


def _grader(deck, **kw):
    params = dict(check_timeout=60, bench_timeout=60, worker_id="test")
    params.update(kw)
    return KernelBenchGrader(deck["repo"], deck["subdir"], deck["problems_dir"], deck["commit"],
                             expected_gpu=None, **params)


def _clean(deck) -> bool:
    out = subprocess.run(["git", "-C", str(deck["repo"]), "status", "--porcelain", "--untracked-files=all"],
                         capture_output=True, text=True).stdout
    return out.strip() == ""


def test_pass_and_score(fake_deck):
    r = _grader(fake_deck).grade("01_toy", GOOD_SOLUTION.format(score=0.25))
    assert r.correct and r.scored and r.peak_fraction == pytest.approx(0.25)
    assert r.shape_fractions == [0.25, 0.25, 0.25]
    assert r.fail_stage is None and "PASS" in r.check_tail
    assert _clean(fake_deck), "deck must be restored after grading"


@pytest.mark.parametrize("code,stage", [
    ("class Model:\n    def forward(self, x):\n        return x * 3\n", c.STAGE_CHECK),
    ("class Model(:\n", c.STAGE_IMPORT),
    ("f = 'torch.nn.functional.linear'\nclass Model:\n    def forward(self, x): return 2 * x\n",
     c.STAGE_FORBIDDEN),
])
def test_failures(fake_deck, code, stage):
    r = _grader(fake_deck).grade("01_toy", code)
    assert not r.correct and r.fail_stage == stage and r.peak_fraction is None
    assert _clean(fake_deck)


def test_secrets_not_visible_to_candidate(fake_deck, monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "super-secret")
    monkeypatch.setenv("MY_SECRET_TOKEN", "also-secret")
    r = _grader(fake_deck).grade("01_toy", GOOD_SOLUTION.format(score=0.2))
    assert r.correct and "LEAK" not in r.check_tail
    env = scrubbed_env({"WANDB_API_KEY": "x", "HF_TOKEN": "y", "GITHUB_PAT": "z", "PATH": "/bin"})
    assert env == {"PATH": "/bin"}


def test_timeout_kills_candidate(fake_deck):
    code = "import time\ntime.sleep(60)\nclass Model:\n    def forward(self, x): return 2 * x\n"
    t0 = time.monotonic()
    r = _grader(fake_deck, check_timeout=1.0).grade("01_toy", code)
    assert time.monotonic() - t0 < 20
    assert r.fail_stage == c.STAGE_TIMEOUT
    assert _clean(fake_deck)


def test_tamper_detected_and_restored(fake_deck):
    code = ("from pathlib import Path\nPath('benchmark.py').write_text('print(\"peak_fraction: 0.99\")')\n"
            "class Model:\n    def forward(self, x): return 2 * x\n")
    r = _grader(fake_deck).grade("01_toy", code)
    assert r.fail_stage == c.STAGE_TAMPER and not r.correct
    assert _clean(fake_deck)


def test_candidate_printed_score_markers_are_rejected(fake_deck):
    code = "print('peak_fraction: 0.99')\n" + GOOD_SOLUTION.format(score=0.2)
    r = _grader(fake_deck).grade("01_toy", code)
    assert r.correct and not r.scored and r.fail_stage == c.STAGE_BENCHMARK


def test_guards(fake_deck):
    assert _grader(fake_deck).grade("../etc", "x").fail_stage == c.STAGE_INTEGRITY
    with pytest.raises(RuntimeError, match="expected"):
        KernelBenchGrader(fake_deck["repo"], fake_deck["subdir"], fake_deck["problems_dir"], "0" * 40,
                          expected_gpu=None, check_timeout=1, bench_timeout=1)
    with pytest.raises(RuntimeError, match="GPU"):
        _grader(fake_deck, check_timeout=1).__class__(
            fake_deck["repo"], fake_deck["subdir"], fake_deck["problems_dir"], fake_deck["commit"],
            expected_gpu="RTX PRO 6000", check_timeout=1, bench_timeout=1)


def test_toolkit_missing_is_reported_as_toolchain(fake_deck, monkeypatch):
    import looppp.grade as grade_mod
    monkeypatch.setattr(grade_mod, "find_cuda_home", lambda roots=(): None)
    code = ("raise OSError('CUDA_HOME environment variable is not set. Please set it to your CUDA install root.')\n"
            "class Model:\n    def forward(self, x): return 2 * x\n")
    r = _grader(fake_deck).grade("01_toy", code)
    assert r.fail_stage == c.STAGE_TOOLCHAIN and "no CUDA toolkit" in r.fail_reason


def test_built_toolkit_is_passed_to_grading(fake_deck, tmp_path, monkeypatch):
    from looppp import cudatk
    for var in ("CUDA_HOME", "CUDA_PATH"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(cudatk.shutil, "which", lambda name: None)
    monkeypatch.setattr(cudatk, "_valid_home", lambda p: bool(p) and (__import__("pathlib").Path(p) / "bin" / "nvcc").exists()
                        and str(p) != "/usr/local/cuda")
    home = tmp_path / "tk" / "cuda-13.0"
    (home / "bin").mkdir(parents=True)
    (home / "bin" / "nvcc").write_text("")
    (home / cudatk.MARKER).write_text("x")
    code = ("import os\nprint('CUDA_HOME_SEEN=' + os.environ.get('CUDA_HOME', ''))\n"
            "class Model:\n    def forward(self, x): return 2 * x\n")
    g = _grader(fake_deck, toolkit_root=tmp_path / "tk")
    assert g.cuda_home() == str(home) and g.describe()["cuda_home"] == str(home)
    r = g.grade("01_toy", code)
    assert f"CUDA_HOME_SEEN={home}" in r.check_tail
