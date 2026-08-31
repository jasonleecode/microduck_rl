"""--device backend resolution (mjlab_microduck.device) must be honest:

- "metal" maps inference to CoreMLExecutionProvider (Apple GPU/ANE) and
  training to CPU (warp-lang has no Metal backend) — never silently to CUDA.
- Explicit requests must raise when the execution provider is missing from
  the onnxruntime build instead of letting onnxruntime fall back to CPU.
- train_cli must strip --device from argv before forwarding to mjlab (mjlab's
  tyro parser would reject the unknown flag).
"""

import sys

import onnxruntime as ort
import pytest

from mjlab_microduck.device import apply_train_device, resolve_ort_providers
from mjlab_microduck.train_cli import _extract_device_flag

AVAILABLE = ort.get_available_providers()


def test_cpu_provider():
    assert resolve_ort_providers("cpu") == ["CPUExecutionProvider"]


@pytest.mark.skipif("CoreMLExecutionProvider" not in AVAILABLE, reason="macOS wheel only")
def test_metal_maps_to_coreml():
    providers = resolve_ort_providers("metal")
    assert providers[0] == "CoreMLExecutionProvider"
    assert providers[-1] == "CPUExecutionProvider"


def test_cuda_raises_without_cuda_ep():
    if "CUDAExecutionProvider" in AVAILABLE:
        assert resolve_ort_providers("cuda")[0] == "CUDAExecutionProvider"
    else:
        with pytest.raises(RuntimeError, match="CUDAExecutionProvider"):
            resolve_ort_providers("cuda")


def test_auto_always_has_cpu_fallback():
    providers = resolve_ort_providers("auto")
    assert providers[-1] == "CPUExecutionProvider"
    assert providers[0] in AVAILABLE


def test_auto_prefers_coreml_on_macos():
    if sys.platform == "darwin" and "CoreMLExecutionProvider" in AVAILABLE:
        assert resolve_ort_providers("auto")[0] == "CoreMLExecutionProvider"


def test_unknown_device_rejected():
    with pytest.raises(ValueError, match="unknown --device"):
        resolve_ort_providers("tpu")
    with pytest.raises(ValueError, match="unknown --device"):
        apply_train_device("tpu")


def test_train_cpu_forces_cpu_mode(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert apply_train_device("cpu") == "cpu"
    assert os_environ_cuda_visible() == ""


def test_train_metal_is_cpu_with_warning(monkeypatch, capsys):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert apply_train_device("metal") == "cpu"
    assert os_environ_cuda_visible() == ""
    assert "no Metal backend" in capsys.readouterr().out


def os_environ_cuda_visible():
    import os

    return os.environ.get("CUDA_VISIBLE_DEVICES")


def test_extract_device_flag():
    device, rest = _extract_device_flag(["Mjlab-Task", "--device", "metal", "--env.scene.num-envs", "8"])
    assert device == "metal"
    assert rest == ["Mjlab-Task", "--env.scene.num-envs", "8"]

    device, rest = _extract_device_flag(["--device=cpu", "Mjlab-Task"])
    assert device == "cpu"
    assert rest == ["Mjlab-Task"]

    device, rest = _extract_device_flag(["Mjlab-Task"])
    assert device == "auto"
    assert rest == ["Mjlab-Task"]

    with pytest.raises(ValueError, match="requires a value"):
        _extract_device_flag(["--device"])


def test_main_writes_stripped_argv_back(monkeypatch):
    """mjlab's tyro parser reads sys.argv — --device must be gone from it,
    or every --device run dies at argument parsing (regression)."""
    import mjlab_microduck.train_cli as train_cli

    monkeypatch.setattr(sys, "argv", ["train", "Mjlab-Task", "--device", "metal", "--hf-jobs"])
    monkeypatch.setattr(train_cli, "main", train_cli.main)  # no-op, clarity
    import mjlab_microduck.hf_jobs as hf_jobs

    captured = {}

    def fake_submit(argv):
        captured["argv"] = argv
        captured["sys_argv"] = list(sys.argv)
        return 0

    monkeypatch.setattr(hf_jobs, "submit", fake_submit)
    assert train_cli.main() == 0
    assert "--device" not in captured["sys_argv"]
    assert "metal" not in captured["sys_argv"]
    assert captured["argv"] == ["Mjlab-Task"]
