import json
import subprocess

import pytest

from looppp.archive import Archive
from looppp.traces import TraceReplayError, replay_solution


def _tool(name, **inp):
    return json.dumps({"type": "assistant", "message": {"content": [{"type": "tool_use", "name": name, "input": inp}]}})


def test_replay_claude_and_opencode_styles():
    lines = [
        _tool("Write", file_path="/w/problems/x/solution.py", content="a = 1\nb = 2\n"),
        _tool("Edit", file_path="/w/problems/x/solution.py", old_string="a = 1", new_string="a = 10"),
        _tool("Edit", file_path="/w/problems/x/other.py", old_string="b = 2", new_string="b = 99"),
        _tool("edit", filePath="/w/solution.py", oldString="b = 2", newString="b = 20"),
        _tool("Edit", file_path="/w/solution.py", old_string="missing", new_string="ignored"),
        _tool("MultiEdit", file_path="/w/solution.py",
              edits=[{"old_string": "10", "new_string": "11"}, {"old_string": "20", "new_string": "21"}]),
        "not json",
    ]
    assert replay_solution(lines) == "a = 11\nb = 21\n"


def test_replay_requires_write():
    with pytest.raises(TraceReplayError):
        replay_solution([_tool("Edit", file_path="solution.py", old_string="a", new_string="b")])


def test_archive_only_improves(tmp_path):
    a = Archive(tmp_path / "archive")
    assert a.best("p") is None
    assert a.maybe_update("p", "v1", {"peak_fraction": 0.2})
    assert not a.maybe_update("p", "worse", {"peak_fraction": 0.1})
    assert not a.maybe_update("p", "unscored", {"peak_fraction": None})
    assert a.maybe_update("p", "v2", {"peak_fraction": 0.3})
    code, meta = a.best("p")
    assert code == "v2" and meta["peak_fraction"] == 0.3


def test_archive_git_commit_only_archive_paths(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git = lambda *args: subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True)
    git("init", "-q")
    git("config", "user.email", "t@t")
    git("config", "user.name", "t")
    (repo / "unrelated.txt").write_text("staged but not archive")
    git("add", "unrelated.txt")
    Archive(repo / "archive", git_commit=True).maybe_update("p", "code", {"peak_fraction": 0.4})
    files = git("show", "--name-only", "--format=", "HEAD").stdout.split()
    assert sorted(files) == ["archive/p/best.json", "archive/p/best.py"]
    assert "A  unrelated.txt" in git("status", "--porcelain").stdout


def test_needs_cuda_toolkit_detection():
    from looppp.traces import needs_cuda_toolkit
    assert needs_cuda_toolkit("from torch.utils.cpp_extension import load\nmod = load(name='x', sources=[])")
    assert needs_cuda_toolkit("from torch.utils.cpp_extension import load_inline")
    assert not needs_cuda_toolkit("import triton\nimport triton.language as tl\n@triton.jit\ndef k(): pass")
