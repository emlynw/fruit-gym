"""
optimized_scene.py  •  tested on MuJoCo 3.3.3

Keys
----
S   randomise strawberry mesh scale (spec → recompile, leak-free)
P   randomise vine root positions   (mjModel in-place, instant)
Q   or  Esc  quit
"""
import time
from pathlib import Path
from typing import Sequence

import cv2
import mujoco
import numpy as np
import psutil


# --------------------------------------------------------------------------- #
#  Helper functions                                                            #
# --------------------------------------------------------------------------- #
def randomise_mesh_scale_in_spec(spec: mujoco.MjSpec, prefixes: Sequence[str]):
    """Multiply `scale` of every mesh whose name starts with a prefix."""
    factor = np.random.uniform(0.5, 2.0)
    print(f"\n[Scale] ×{factor:.3f}")
    for mesh in spec.meshes:
        if any(mesh.name.startswith(p) for p in prefixes):
            if mesh.scale is None:
                mesh.scale = np.ones(3)
            mesh.scale *= factor
            print(f"  {mesh.name}: {np.round(mesh.scale, 3)}")


def randomise_vine_positions(model: mujoco.MjModel, vine_ids: np.ndarray):
    lo = np.array([0.00, -1.0, 0.4])
    hi = np.array([1.00,  1.0, 0.8])
    new_pos = np.random.uniform(lo, hi, (len(vine_ids), 3))
    model.body_pos[vine_ids, :] = new_pos
    for bid, pos in zip(vine_ids, new_pos):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        print(f"  {name}: {np.round(pos, 3)}")


def move_unique_assets(dest: mujoco.MjSpec, src: mujoco.MjSpec):
    """Move (don’t copy) assets from *src* → *dest* unless name already there."""
    for coll in ("meshes", "textures", "materials", "skins", "hfields"):
        dlist, slist = getattr(dest, coll), getattr(src, coll)
        existing = {a.name for a in dlist if a.name}
        newly_moved = [a for a in slist if a.name and a.name not in existing]
        dlist.extend(newly_moved)
        for a in newly_moved:          # remove to avoid double-free later
            slist.remove(a)


# --------------------------------------------------------------------------- #
#  Build the scene                                                            #
# --------------------------------------------------------------------------- #
def build_scene(n_vines: int = 8):
    """Return (spec, list_of_root_body_names)."""
    scene_spec = mujoco.MjSpec.from_file("scene.xml")
    root_names = []

    assets_moved = False
    for i in range(n_vines):
        vine_spec = mujoco.MjSpec.from_file("strawb.xml")

        if not assets_moved:                       # first iteration only
            move_unique_assets(scene_spec, vine_spec)
            assets_moved = True

        mount = scene_spec.worldbody.add_frame(name=f"vine_mount_{i}")
        root  = mount.attach_body(vine_spec.worldbody.bodies[0],
                                  suffix=f"_{i}")
        root_names.append(root.name)
        # vine_spec goes out of scope here — any leftover RAM is freed by GC

    return scene_spec, root_names


# --------------------------------------------------------------------------- #
#  Main loop                                                                  #
# --------------------------------------------------------------------------- #
def main():
    # ---------- build & compile ----------------------------------------------
    scene_spec, root_names = build_scene(8)
    model  = scene_spec.compile()
    data   = mujoco.MjData(model)

    vine_ids = np.array([model.body(n).id for n in root_names])
    mesh_prefixes = ["strawberry", "strawberry_leaves", "strawberry_collision"]

    # ---------- renderer & user camera (same as your original) ---------------
    renderer = mujoco.Renderer(model, 480, 480)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.lookat[:] = (0.5, 0.0, 0.6)
    cam.distance  = 1.0
    cam.elevation = -20
    cam.azimuth   = 90

    proc = psutil.Process()
    print("\nControls:  S = scale   |   P = position   |   Q / Esc = quit\n")

    try:
        while True:
            tic = time.time()
            mujoco.mj_step(model, data)

            renderer.update_scene(data, camera=cam)
            bgr = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)

            mem = proc.memory_info().rss / (1024 * 1024)
            cv2.putText(bgr, f"Memory: {mem:6.1f} MB", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            cv2.imshow("MuJoCo Simulation", bgr)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):                        # quit
                break

            elif key == ord("s"):                            # random scale
                randomise_mesh_scale_in_spec(scene_spec, mesh_prefixes)
                model, data = scene_spec.recompile(model, data)   # in-place
                renderer.close(); renderer = mujoco.Renderer(model, 480, 480)

            elif key == ord("p"):                            # random position
                randomise_vine_positions(model, vine_ids)
                mujoco.mj_forward(model, data)

            # keep real-time
            dt = model.opt.timestep - (time.time() - tic)
            if dt > 0:
                time.sleep(dt)

    finally:
        renderer.close()
        cv2.destroyAllWindows()
        print("Exited cleanly.")


if __name__ == "__main__":
    for fname in ("scene.xml", "strawb.xml"):
        if not Path(fname).exists():
            raise FileNotFoundError(fname)
    main()
