# Pretrained MicroDuck policies

Downloaded deployment policies (61D obs / 14D action contract), for use with
`scripts/infer_policy.py`:

```bash
uv run mjpython scripts/infer_policy.py \
  --walking policies/alpha/alpha_walking.onnx \
  --standing policies/alpha/alpha_stand.onnx \
  --ground-pick policies/alpha/alpha_ground_pick.onnx \
  --kick-left policies/alpha/ball_kick_left.onnx \
  --kick-right policies/alpha/ball_kick_right.onnx \
  --new-cmd-obs --device metal
```

## Provenance

- `alpha/` — the official policies shipped with the robot runtime, from
  [pollen-robotics/microduck](https://github.com/pollen-robotics/microduck)
  `policies/` directory (walking, standing, ground pick, ball kick left/right).
- `rough_walk_e/` — community walking policy tuned for rough terrain, from
  [RemiFabre/microduck-rough-walk-e](https://huggingface.co/RemiFabre/microduck-rough-walk-e)
  on Hugging Face (see `manifest.json` for its command ranges).

These are binary inputs for inference demos, which is why `policies/**` is
exempted from the `*.onnx` rule in `.gitignore`. Training outputs still do not
belong in the repo.
