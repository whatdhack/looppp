import json

from looppp import envcheck
from looppp.envcheck import gpu_matches, gpu_probe, parse_procfs_information

TORCH_OK = json.dumps({"torch": "2.11.0", "cuda": "13.0", "available": True, "count": 1,
                       "name": "NVIDIA RTX PRO 6000 Blackwell Server Edition", "capability": [12, 0]})


def _fake_sh(nvidia=(False, ""), torch=(True, TORCH_OK)):
    def sh(cmd, timeout):
        return nvidia if cmd.startswith("nvidia-smi") else torch
    return sh


def test_falls_back_to_torch_when_nvidia_smi_missing(monkeypatch):
    monkeypatch.setattr(envcheck.shutil, "which", lambda name: None)
    p = gpu_probe("python", sh=_fake_sh())
    assert p["source"] == "torch"
    assert p["name"] == "NVIDIA RTX PRO 6000 Blackwell Server Edition"
    assert p["nvidia_smi"] == "not installed"
    assert gpu_matches(p["name"], "RTX PRO 6000")


def test_prefers_nvidia_smi(monkeypatch):
    monkeypatch.setattr(envcheck.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    p = gpu_probe("python", sh=_fake_sh(nvidia=(True, "NVIDIA RTX PRO 6000 Blackwell, 580.1, 97887 MiB, 600 W")))
    assert (p["source"], p["name"]) == ("nvidia-smi", "NVIDIA RTX PRO 6000 Blackwell")


def test_nvidia_smi_driver_error_is_not_a_name(monkeypatch):
    monkeypatch.setattr(envcheck.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    sh = _fake_sh(nvidia=(True, "NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver"),
                  torch=(False, "ModuleNotFoundError: No module named 'torch'"))
    p = gpu_probe("python", sh=sh)
    assert p["name"] == "" and "ModuleNotFoundError" in p["torch"]
    assert not gpu_matches(p["name"], "RTX PRO 6000") and gpu_matches(p["name"], None)


def test_parse_procfs():
    text = "Model: \t\t NVIDIA RTX PRO 6000 Blackwell Workstation Edition\nIRQ: 16\n"
    assert parse_procfs_information(text) == "NVIDIA RTX PRO 6000 Blackwell Workstation Edition"
    assert parse_procfs_information("IRQ: 16") == ""
