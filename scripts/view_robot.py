"""Open a MuJoCo scene in the interactive viewer.

Usage: uv run python scripts/view_robot.py [scene.xml]

NOTE: use plain `python`, NOT `mjpython` — mjpython moves Python off the main
thread, and recent macOS/GLFW refuse to create the window from a worker thread
("NSWindow should only be instantiated on the main thread!"). Plain python runs
on the main thread, which is exactly what the Cocoa window needs.
"""

import sys

import mujoco
import mujoco.viewer

xml = sys.argv[1] if len(sys.argv) > 1 else "src/mjlab_microduck/robot/microduck/scene.xml"
model = mujoco.MjModel.from_xml_path(xml)
data = mujoco.MjData(model)
print(f"Loaded {xml}: {model.nbody} bodies, {model.njnt} joints")
mujoco.viewer.launch(model, data)
