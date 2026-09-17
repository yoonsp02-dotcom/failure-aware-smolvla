"""Check the local LeRobot, CUDA, and headless MuJoCo setup."""

from __future__ import annotations

import os
import sys

import lerobot
import mujoco
import torch
import transformers


def main() -> None:
    print(f"Python: {sys.version.split()[0]}")
    print(f"LeRobot: {lerobot.__version__}")
    print(f"Transformers: {transformers.__version__}")
    print(f"PyTorch: {torch.__version__}")
    print(f"MUJOCO_GL: {os.environ.get('MUJOCO_GL', '<unset>')}")
    print(f"PYOPENGL_PLATFORM: {os.environ.get('PYOPENGL_PLATFORM', '<unset>')}")

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access CUDA.")

    device = torch.cuda.get_device_name(0)
    result = torch.ones((32, 32), device="cuda").sum().item()
    print(f"CUDA: {device} (tensor test={result:.0f})")

    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><geom type="plane" size="1 1 .1"/></worldbody></mujoco>'
    )
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=64, width=64)
    try:
        renderer.update_scene(data)
        image = renderer.render()
        print(f"MuJoCo EGL: {image.shape} {image.dtype}")
    finally:
        renderer.close()


if __name__ == "__main__":
    main()
