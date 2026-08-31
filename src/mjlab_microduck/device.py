"""Accelerator backend selection shared by training and inference.

The physics engine (MuJoCo Warp, via NVIDIA Warp) has exactly two backends:
CUDA and CPU. The warp-lang macOS wheels ship WITHOUT a Metal backend, so
Apple Silicon training always runs physics on CPU. Inference (onnxruntime)
can use the Apple GPU / Neural Engine through the CoreML execution provider,
exposed here as "metal".

Both entry points accept the same `--device {auto,cuda,cpu,metal}` flag:
  - `uv run train ... --device cpu`          (mjlab would not know the flag)
  - `uv run scripts/infer_policy.py ... --device metal`
"""

from __future__ import annotations

import os
import sys

DEVICE_CHOICES = ("auto", "cuda", "cpu", "metal")

_ORT_PROVIDERS = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    # CoreML dispatches to the Apple Neural Engine / GPU (Metal) at runtime.
    "metal": "CoreMLExecutionProvider",
}


def apply_train_device(device: str = "auto") -> str:
    """Resolve the training backend and configure the environment for mjlab.

    mjlab picks its device from CUDA_VISIBLE_DEVICES alone (empty = CPU,
    anything else = cuda:<local_rank>), so that variable is the whole
    mechanism. Must be called BEFORE mjlab/torch touch the GPU — train_cli
    calls it before importing mjlab.scripts.train.

    Returns the resolved backend: "cuda" or "cpu".
    """
    if device not in DEVICE_CHOICES:
        raise ValueError(f"unknown --device {device!r}, expected one of {DEVICE_CHOICES}")

    if device == "metal":
        print(
            "[device] warp-lang has no Metal backend — training physics runs on CPU. "
            "(Metal acceleration applies to inference: scripts/infer_policy.py --device metal.)"
        )
        device = "cpu"

    if device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "--device cuda requested but torch sees no CUDA GPU "
                f"(torch.cuda.is_available()={torch.cuda.is_available()}). "
                "Use --device cpu on machines without an NVIDIA GPU."
            )
        # Leave CUDA_VISIBLE_DEVICES untouched: mjlab maps it to cuda:<local_rank>.
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    return device


def resolve_ort_providers(device: str = "auto") -> list[str]:
    """Map a device choice to an onnxruntime provider list (first match wins).

    Explicit choices raise if the execution provider is missing from the
    onnxruntime build, instead of letting onnxruntime silently fall back to
    CPU (a CUDA request needs the onnxruntime-gpu wheel; CoreML only ships in
    the macOS build).
    """
    if device not in DEVICE_CHOICES:
        raise ValueError(f"unknown --device {device!r}, expected one of {DEVICE_CHOICES}")

    import onnxruntime as ort

    available = ort.get_available_providers()

    if device == "auto":
        if sys.platform == "darwin" and "CoreMLExecutionProvider" in available:
            device = "metal"
        elif "CUDAExecutionProvider" in available:
            device = "cuda"
        else:
            device = "cpu"

    provider = _ORT_PROVIDERS[device]
    if provider not in available:
        raise RuntimeError(
            f"--device {device} requires {provider}, which is not in this onnxruntime "
            f"build (available: {available}). CUDA needs the onnxruntime-gpu package; "
            "metal (CoreML) only exists in the macOS wheel."
        )
    if provider == "CPUExecutionProvider":
        return [provider]
    return [provider, "CPUExecutionProvider"]
