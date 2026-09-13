"""Provide a CUDA toolkit (nvcc + headers) for torch.utils.cpp_extension on a box that
only has the CUDA *runtime* (e.g. molab: pip torch, no nvcc, CUDA_HOME unset).

NVIDIA ships nvcc and headers as pip wheels. CUDA 13 wheels install into one
``site-packages/nvidia/cu13/`` tree; CUDA 12 wheels use one directory per
component (``nvidia/cuda_nvcc``, ``nvidia/cuda_runtime``, ``nvidia/cublas`` ...).
Either way we assemble a symlink directory shaped like a normal CUDA install
(bin/, nvvm/, include/, lib64/ with unversioned ``libX.so`` links) and point
CUDA_HOME at it.

Only the missing compiler pieces are installed, with ``--no-deps`` and pinned to
torch's CUDA major.minor, so the CUDA libraries torch itself pins are never
upgraded underneath it.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

MARKER = ".looppp-cuda-home"


@dataclass
class ToolkitReport:
    ok: bool
    cuda_home: str | None = None
    source: str = ""                 # env | PATH | /usr/local/cuda | pip
    torch_cuda: str | None = None
    nvcc_version: str = ""
    installed: list[str] = field(default_factory=list)
    log: list[str] = field(default_factory=list)


def _run(cmd: list[str], timeout: float = 120) -> tuple[int, str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout + r.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as e:
        return 1, f"{type(e).__name__}: {e}"


def _valid_home(p: Path | str | None) -> bool:
    return bool(p) and (Path(p) / "bin" / "nvcc").exists()


def nvcc_release(cuda_home: Path | str) -> str:
    rc, out = _run([str(Path(cuda_home) / "bin" / "nvcc"), "--version"], 60)
    m = re.search(r"release (\d+\.\d+)", out)
    return m.group(1) if rc == 0 and m else ""


def find_cuda_home(search_roots: list[Path] | tuple[Path, ...] = ()) -> tuple[str, str] | None:
    """Existing toolkit: $CUDA_HOME/$CUDA_PATH, nvcc on PATH, /usr/local/cuda, then any
    CUDA_HOME looppp assembled earlier under *search_roots* (newest first)."""
    for var in ("CUDA_HOME", "CUDA_PATH"):
        if _valid_home(os.environ.get(var)):
            return os.environ[var], "env"
    nvcc = shutil.which("nvcc")
    if nvcc:
        home = Path(os.path.realpath(nvcc)).parent.parent
        if _valid_home(home):
            return str(home), "PATH"
    if _valid_home("/usr/local/cuda"):
        return "/usr/local/cuda", "/usr/local/cuda"
    built = [m.parent for root in search_roots for m in Path(root).glob(f"cuda-*/{MARKER}")
             if _valid_home(m.parent)]
    if built:
        return str(max(built, key=lambda p: (p / MARKER).stat().st_mtime)), "looppp-built"
    return None


def torch_cuda_version(python: str = sys.executable) -> str | None:
    rc, out = _run([python, "-c", "import torch; print(torch.version.cuda or '')"], 180)
    ver = out.strip().splitlines()[-1] if rc == 0 and out.strip() else ""
    return ver if re.fullmatch(r"\d+\.\d+(\.\d+)?", ver) else None


def component_packages(cuda_version: str) -> list[str]:
    major = int(cuda_version.split(".")[0])
    if major >= 13:
        return ["nvidia-cuda-nvcc", "nvidia-cuda-crt", "nvidia-nvvm", "nvidia-cuda-runtime"]
    if major == 12:
        return ["nvidia-cuda-nvcc-cu12", "nvidia-cuda-runtime-cu12"]
    raise ValueError(f"unsupported CUDA version {cuda_version}")


def pip_specs(cuda_version: str, installed: dict[str, str], upgrade_compiler: bool = False) -> list[str]:
    """Compiler pieces (nvcc, crt, nvvm): newest release within torch's CUDA *major* version. Old
    minors can fail against a newer host glibc (CUDA 13.0 crt vs glibc 2.43 `rsqrt`), and CUDA
    guarantees minor-version compatibility, so torch only warns about a minor mismatch.
    Runtime headers: torch's exact major.minor, and only if torch did not already install them."""
    major, minor = (int(x) for x in cuda_version.split(".")[:2])
    specs = []
    for pkg in component_packages(cuda_version):
        if "runtime" in pkg:
            if pkg not in installed:
                specs.append(f"{pkg}>={major}.{minor},<{major}.{minor + 1}")
        elif pkg not in installed or upgrade_compiler:
            specs.append(f"{pkg}>={major}.0,<{major + 1}")
    return specs


_PROBE_KERNEL = """#include <cstdio>
#include <cuda.h>

__global__ void looppp_probe(int n, const float* x, float* y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] += x[i];
}

int main() { int d = 0; return cudaGetDeviceCount(&d) == 0 ? 0 : 0; }
"""


def compile_check(cuda_home: Path | str, workdir: Path) -> tuple[bool, str]:
    """Compile and link a tiny CUDA program: proves nvcc, crt headers, host compiler and libcudart line up."""
    workdir.mkdir(parents=True, exist_ok=True)
    src, exe = workdir / "looppp_probe.cu", workdir / "looppp_probe"
    src.write_text(_PROBE_KERNEL)
    home = Path(cuda_home)
    rc, out = _run([str(home / "bin" / "nvcc"), "-O1", str(src), "-o", str(exe), f"-L{home / 'lib64'}", "-lcudart"], 300)
    return rc == 0 and exe.exists(), out[-1500:]


def installed_versions(python: str, names: list[str]) -> dict[str, str]:
    code = ("import json, importlib.metadata as m\nout = {}\n"
            f"for n in {names!r}:\n    try:\n        out[n] = m.version(n)\n    except m.PackageNotFoundError:\n        pass\n"
            "print(json.dumps(out))")
    rc, out = _run([python, "-c", code], 60)
    try:
        return json.loads(out.strip().splitlines()[-1]) if rc == 0 else {}
    except (json.JSONDecodeError, IndexError):
        return {}


def purelib(python: str = sys.executable) -> Path:
    rc, out = _run([python, "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"], 60)
    return Path(out.strip().splitlines()[-1])


def build_cuda_home(nvidia_dir: Path, dest: Path) -> Path:
    """Assemble a CUDA_HOME from pip-installed NVIDIA wheels under *nvidia_dir* (site-packages/nvidia)."""
    nvidia_dir, dest = Path(nvidia_dir), Path(dest)
    if (nvidia_dir / "cu13" / "bin" / "nvcc").exists():
        sources = [nvidia_dir / "cu13"]
    else:
        sources = sorted(d for d in nvidia_dir.iterdir() if d.is_dir() and not d.name.startswith("__"))
        # put the compiler package first so its bin/ and nvvm/ win
        sources.sort(key=lambda d: 0 if (d / "bin" / "nvcc").exists() else 1)
    if not any((s / "bin" / "nvcc").exists() for s in sources):
        raise FileNotFoundError(f"no nvcc under {nvidia_dir}")

    if dest.exists():
        if not (dest / MARKER).exists():
            raise FileExistsError(f"{dest} exists and was not created by looppp; refusing to replace it")
        shutil.rmtree(dest)
    (dest / "include").mkdir(parents=True)
    (dest / "lib64").mkdir()

    for src in sources:
        for sub in ("bin", "nvvm"):
            if (src / sub).is_dir() and not (dest / sub).exists():
                (dest / sub).symlink_to(src / sub, target_is_directory=True)
        for sub, out in (("include", "include"), ("lib", "lib64"), ("lib64", "lib64")):
            if not (src / sub).is_dir():
                continue
            for entry in (src / sub).iterdir():
                if entry.name in ("__init__.py", "__pycache__"):
                    continue
                target = dest / out / entry.name
                if not target.exists() and not target.is_symlink():
                    target.symlink_to(entry, target_is_directory=entry.is_dir())

    for entry in list((dest / "lib64").iterdir()):
        m = re.match(r"^(lib.+\.so)(\.\d+)+$", entry.name)
        if m and not (dest / "lib64" / m.group(1)).exists():
            (dest / "lib64" / m.group(1)).symlink_to(entry.resolve())

    (dest / MARKER).write_text(f"built from {nvidia_dir}\n")
    return dest


def activate(cuda_home: str | Path) -> None:
    """Make CUDA_HOME visible to this process and to grading subprocesses (they copy os.environ)."""
    home = str(cuda_home)
    os.environ["CUDA_HOME"] = home
    bin_dir = os.path.join(home, "bin")
    if bin_dir not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")


def ensure_cuda_toolkit(python: str = sys.executable, build_root: Path = Path(".looppp"),
                        install: bool = True) -> ToolkitReport:
    found = find_cuda_home([Path(build_root)])
    if found and found[1] == "looppp-built":
        ok, _ = compile_check(found[0], Path(build_root) / "cuda-probe")
        found = found if ok else None  # rebuild below if the earlier assembly no longer works
    if found:
        rep = ToolkitReport(True, found[0], found[1], torch_cuda_version(python), nvcc_release(found[0]))
        activate(found[0])
        return rep

    rep = ToolkitReport(False, source="pip", torch_cuda=torch_cuda_version(python))
    if not rep.torch_cuda:
        rep.log.append("torch has no CUDA build (torch.version.cuda is empty); nothing to match nvcc against")
        return rep

    names = component_packages(rep.torch_cuda)
    have = installed_versions(python, names)
    rep.log.append(f"torch CUDA {rep.torch_cuda}; already installed: {have or 'none'}")

    def pip_install(specs: list[str]) -> bool:
        cmd = [python, "-m", "pip", "install", "--no-deps", "-q", "-U", *specs]
        rc, out = _run(cmd, 1200)
        if rc != 0 and "No module named pip" in out and shutil.which("uv"):
            cmd = ["uv", "pip", "install", "--python", python, "--no-deps", "-q", "-U", *specs]
            rc, out = _run(cmd, 1200)
        rep.log.append(f"$ {' '.join(cmd)}\n{out[-1500:]}".rstrip())
        if rc == 0:
            rep.installed += specs
        return rc == 0

    specs = pip_specs(rep.torch_cuda, have)
    if specs:
        if not install:
            rep.log.append(f"missing: {', '.join(specs)} (install not requested)")
            return rep
        if not pip_install(specs):
            return rep

    home = Path(build_root) / f"cuda-{rep.torch_cuda}"
    for attempt in (1, 2):
        try:
            build_cuda_home(purelib(python) / "nvidia", home)
        except (OSError, ValueError) as e:
            rep.log.append(f"could not assemble CUDA_HOME: {type(e).__name__}: {e}")
            return rep
        rep.cuda_home, rep.nvcc_version = str(home), nvcc_release(home)
        if not rep.nvcc_version:
            rep.log.append("nvcc is present but `nvcc --version` failed")
            return rep
        if rep.nvcc_version.split(".")[0] != rep.torch_cuda.split(".")[0]:
            rep.log.append(f"nvcc {rep.nvcc_version} and torch CUDA {rep.torch_cuda} have different major versions")
            return rep
        ok, out = compile_check(home, Path(build_root) / "cuda-probe")
        if ok:
            break
        rep.log.append(f"nvcc {rep.nvcc_version} could not build a test kernel:\n{out}")
        if attempt == 2 or not install or not pip_install(pip_specs(rep.torch_cuda, have, upgrade_compiler=True)):
            return rep
        rep.log.append("retrying with the newest compiler in this CUDA major version")

    if rep.nvcc_version != ".".join(rep.torch_cuda.split(".")[:2]):
        rep.log.append(f"note: nvcc {rep.nvcc_version} vs torch CUDA {rep.torch_cuda}: same major version, "
                       "torch prints a minor-version warning and builds normally")
    activate(home)
    rep.ok = True
    return rep
