"""`train` entry point: mjlab's trainer, plus `--hf-jobs` remote submission.

This project's [project.scripts] `train` shadows mjlab's so the everyday
command grows one flag:

    uv run train Mjlab-Kick-Flat-MicroDuck --env.scene.num-envs 4096 \
        --agent.max_iterations 4000              # local, exactly as before
    uv run train Mjlab-Kick-Flat-MicroDuck --env.scene.num-envs 4096 \
        --agent.max_iterations 4000 --hf-jobs    # same run, on HF Jobs

Without --hf-jobs, argv is passed to mjlab.scripts.train untouched, except for
one locally-consumed flag mjlab does not know:

    uv run train Mjlab-Kick-Flat-MicroDuck --device cpu    # force CPU training

--device {auto,cuda,cpu,metal} selects the accelerator backend (see
mjlab_microduck.device). On Apple Silicon, "metal" maps to CPU physics —
warp-lang has no Metal backend.
"""

from __future__ import annotations

import sys


def _extract_device_flag(argv: list[str]) -> tuple[str, list[str]]:
    """Pull `--device X` / `--device=X` out of argv. Returns (device, rest)."""
    device = "auto"
    rest: list[str] = []
    skip_next = False
    for i, arg in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        if arg == "--device":
            if i + 1 >= len(argv):
                raise ValueError("--device requires a value: auto, cuda, cpu or metal")
            device = argv[i + 1]
            skip_next = True
        elif arg.startswith("--device="):
            device = arg.split("=", 1)[1]
        else:
            rest.append(arg)
    return device, rest


def main() -> int | None:
    device, argv = _extract_device_flag(sys.argv[1:])
    # mjlab parses sys.argv itself (tyro), so write the stripped argv back.
    sys.argv = [sys.argv[0], *argv]

    if "--hf-jobs" in argv:
        if device != "auto":
            print("[device] ignoring --device for --hf-jobs: remote jobs always run on CUDA")
        from mjlab_microduck.hf_jobs import submit

        return submit([a for a in argv if a != "--hf-jobs"])

    from mjlab_microduck.device import apply_train_device

    apply_train_device(device)

    from mjlab.scripts.train import main as mjlab_train_main

    return mjlab_train_main()


if __name__ == "__main__":
    sys.exit(main())
