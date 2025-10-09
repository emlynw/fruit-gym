#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import mujoco

# ---- tweak if your ordering differs ----
PANDA_HOME   = np.array([0.0, -1.2, 0.0, -2.0, -0.05, 2.49, 0.822], dtype=np.float32)
GRIPPER_HOME = np.array([0.0141, 0.0141], dtype=np.float32)

def save_rgb(img, path):
    Image.fromarray(img).save(path)

def list_cams(model):
    return [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i) or f"camera_{i}"
            for i in range(model.ncam)]

def pick_three_cams(model, requested=None):
    if model.ncam == 0:
        raise RuntimeError("No cameras in scene.xml")
    out = []
    if requested:
        for r in requested[:3]:
            try:
                cid = int(r)
                nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, cid) or f"camera_{cid}"
                out.append((cid, nm))
            except ValueError:
                cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, r)
                if cid == -1:
                    raise RuntimeError(f"Camera '{r}' not found. Available: {list_cams(model)}")
                out.append((cid, r))
    else:
        for cid in range(min(3, model.ncam)):
            nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, cid) or f"camera_{cid}"
            out.append((cid, nm))
    while len(out) < 3:
        out.append(out[-1])
    return out[:3]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", type=Path, default=Path("scene.xml"))
    ap.add_argument("--cameras", nargs="*", default=None)
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--settle", type=int, default=60, help="steps after homing (optional)")
    ap.add_argument("--save-dir", type=Path, default=Path("renders"))
    ap.add_argument("--prefix", type=str, default="frame")
    # if your qpos layout differs, change these slices
    ap.add_argument("--arm-slice", type=str, default="0:7", help="qpos slice for arm, e.g. '0:7'")
    ap.add_argument("--grip-slice", type=str, default="7:9", help="qpos slice for gripper, e.g. '7:9'")
    args = ap.parse_args()

    if not args.xml.exists():
        print(f"Missing XML: {args.xml}", file=sys.stderr); sys.exit(1)

    model = mujoco.MjModel.from_xml_path(str(args.xml))
    data = mujoco.MjData(model)

    # ---- just set qpos directly ----
    arm_sl   = slice(*map(int, args.arm_slice.split(":")))
    grip_sl  = slice(*map(int, args.grip_slice.split(":")))
    data.qpos[arm_sl]  = PANDA_HOME
    data.qpos[grip_sl] = GRIPPER_HOME
    mujoco.mj_forward(model, data)  # apply kinematics

    # optional: settle a bit (contacts, tendons, etc.)
    for _ in range(max(0, args.settle)):
        mujoco.mj_step(model, data)

    # ---- render three cameras ----
    cams = pick_three_cams(model, args.cameras)
    renderer = mujoco.Renderer(model, width=args.width, height=args.height)
    args.save_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for cid, cname in cams:
        renderer.update_scene(data, camera=cid)
        rgb = renderer.render()
        out = args.save_dir / f"{args.prefix}_{cname}.png"
        save_rgb(rgb, out)
        saved.append(out)

    print("Saved:")
    for p in saved:
        print(" ", p)

if __name__ == "__main__":
    main()
