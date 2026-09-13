import os
import stat

import pytest

from looppp import cudatk


def _touch(path, text="", exe=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    if exe:
        path.chmod(path.stat().st_mode | stat.S_IEXEC)


def test_pip_specs_compiler_newest_in_major_runtime_exact():
    assert cudatk.pip_specs("13.0", {"nvidia-cuda-runtime": "13.0.96"}) == [
        "nvidia-cuda-nvcc>=13.0,<14", "nvidia-cuda-crt>=13.0,<14", "nvidia-nvvm>=13.0,<14"]
    assert cudatk.pip_specs("12.8", {}) == ["nvidia-cuda-nvcc-cu12>=12.0,<13", "nvidia-cuda-runtime-cu12>=12.8,<12.9"]
    installed = {"nvidia-cuda-nvcc": "13.0.88", "nvidia-cuda-crt": "13.0.88", "nvidia-nvvm": "13.0.88",
                 "nvidia-cuda-runtime": "13.0.96"}
    assert cudatk.pip_specs("13.0", installed) == []
    assert cudatk.pip_specs("13.0", installed, upgrade_compiler=True) == [
        "nvidia-cuda-nvcc>=13.0,<14", "nvidia-cuda-crt>=13.0,<14", "nvidia-nvvm>=13.0,<14"]
    with pytest.raises(ValueError):
        cudatk.component_packages("11.8")


def test_build_cuda_home_cu13_layout(tmp_path):
    nv = tmp_path / "site" / "nvidia"
    cu = nv / "cu13"
    _touch(cu / "bin" / "nvcc", exe=True)
    _touch(cu / "nvvm" / "bin" / "cicc", exe=True)
    _touch(cu / "include" / "cuda.h")
    _touch(cu / "include" / "crt" / "host_config.h")
    _touch(cu / "include" / "cublas_v2.h")
    _touch(cu / "lib" / "libcudart.so.13")
    _touch(cu / "lib" / "libcublas.so.13")
    home = cudatk.build_cuda_home(nv, tmp_path / "cuda-13.0")

    assert (home / "bin" / "nvcc").exists() and (home / "nvvm" / "bin" / "cicc").exists()
    assert (home / "include" / "cuda.h").exists() and (home / "include" / "crt" / "host_config.h").exists()
    assert os.path.realpath(home / "lib64" / "libcudart.so") == str((cu / "lib" / "libcudart.so.13").resolve())
    assert (home / "lib64" / "libcublas.so").exists()
    # rebuilding our own directory is allowed, replacing a foreign one is not
    cudatk.build_cuda_home(nv, home)
    foreign = tmp_path / "real-cuda"
    foreign.mkdir()
    with pytest.raises(FileExistsError):
        cudatk.build_cuda_home(nv, foreign)


def test_build_cuda_home_cu12_per_component_layout(tmp_path):
    nv = tmp_path / "nvidia"
    _touch(nv / "__init__.py")
    _touch(nv / "cuda_nvcc" / "bin" / "nvcc", exe=True)
    _touch(nv / "cuda_nvcc" / "nvvm" / "bin" / "cicc", exe=True)
    _touch(nv / "cuda_nvcc" / "include" / "crt" / "host_config.h")
    _touch(nv / "cuda_nvcc" / "include" / "__init__.py")
    _touch(nv / "cuda_runtime" / "include" / "cuda.h")
    _touch(nv / "cuda_runtime" / "lib" / "libcudart.so.12")
    _touch(nv / "cublas" / "include" / "cublas_v2.h")
    _touch(nv / "cublas" / "lib" / "libcublas.so.12")
    home = cudatk.build_cuda_home(nv, tmp_path / "cuda-12.8")

    assert (home / "bin" / "nvcc").exists() and (home / "nvvm").exists()
    for header in ("cuda.h", "cublas_v2.h", "crt"):
        assert (home / "include" / header).exists()
    assert not (home / "include" / "__init__.py").exists()
    assert (home / "lib64" / "libcudart.so").exists() and (home / "lib64" / "libcublas.so").exists()


def test_find_cuda_home_env_and_activate(tmp_path, monkeypatch):
    home = tmp_path / "cuda"
    _touch(home / "bin" / "nvcc", exe=True)
    monkeypatch.setenv("CUDA_HOME", str(home))
    assert cudatk.find_cuda_home() == (str(home), "env")
    monkeypatch.setenv("PATH", "/usr/bin")
    cudatk.activate(home)
    assert os.environ["PATH"].split(os.pathsep)[0] == str(home / "bin")
    cudatk.activate(home)
    assert os.environ["PATH"].count(str(home / "bin")) == 1


def test_no_torch_reports_not_ok(tmp_path, monkeypatch):
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.delenv("CUDA_PATH", raising=False)
    monkeypatch.setattr(cudatk, "find_cuda_home", lambda roots=(): None)
    monkeypatch.setattr(cudatk, "torch_cuda_version", lambda python: None)
    rep = cudatk.ensure_cuda_toolkit("python", tmp_path, install=True)
    assert not rep.ok and "no CUDA build" in rep.log[0]
