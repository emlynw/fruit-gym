"""
optimized_scene.py  •  MuJoCo 3.3.x

Adds random per-vine template selection from: strawb.xml, leaves.xml, strawb_fork.xml
"""

import random
import time
from pathlib import Path
from typing import Sequence

import cv2
import mujoco
import numpy as np
import psutil
from PIL import Image
import gc

# ------------------------------------------------------------
# NEW: candidate vine XMLs (relative to cwd)
XML_CHOICES = ("strawb.xml", "leaves.xml", "strawb_fork.xml")
# ------------------------------------------------------------

SKYBOX_DIR = Path.cwd() / "textures" / "skyboxes"
SKYBOX_EXTS = {".png", ".jpg", ".jpeg"}

def find_skybox_images() -> list[Path]:
    if not SKYBOX_DIR.exists():
        return []
    return [p for p in SKYBOX_DIR.iterdir()
            if p.suffix.lower() in SKYBOX_EXTS and p.is_file()]

_CROSS_POS = {"U": (0, 1), "L": (1, 0), "F": (1, 1), "R": (1, 2), "B": (1, 3), "D": (2, 1)}
_STRIP_ORDER = ["F", "B", "U", "D", "R", "L"]

from PIL import Image

def _prepare_face(face_img: Image.Image, face: str) -> Image.Image:
    face_img = face_img.transpose(Image.FLIP_TOP_BOTTOM)
    if face == "U":
        face_img = face_img.rotate(-90, expand=True)
    elif face == "D":
        face_img = face_img.rotate(90, expand=True)
    else:
        face_img = face_img.rotate(180, expand=True)
    return face_img

def _cross_to_strip(img: Image.Image, out_w: int, out_h: int) -> Image.Image:
    tile_in  = img.height // 3
    tile_out = out_w
    strip = Image.new("RGB", (out_w, out_h))
    for i, face in enumerate(_STRIP_ORDER):
        r, c = _CROSS_POS[face]
        crop = img.crop((c * tile_in, r * tile_in, (c + 1) * tile_in, (r + 1) * tile_in))
        crop = _prepare_face(crop, face)
        if tile_in != tile_out:
            crop = crop.resize((tile_out, tile_out), Image.LANCZOS)
        strip.paste(crop, (0, i * tile_out))
    return strip

def apply_texture(model: mujoco.MjModel, context: mujoco.MjrContext, tex_id: int, path: Path):
    try:
        img = Image.open(path).convert("RGB")
        print(f"Applying texture: {path.name}")
    except FileNotFoundError:
        print(f"Warning: texture not found: {path}")
        return
    out_h, out_w = model.tex_height[tex_id], model.tex_width[tex_id]
    if img.size == (out_w, out_h):
        strip = img.transpose(Image.FLIP_TOP_BOTTOM)
    else:
        if img.height % 3 or img.width % 4:
            print(f"{path.name} is neither 1×6 nor 3×4; skipped.")
            return
        strip = _cross_to_strip(img, out_w, out_h)
    pixels = np.asarray(strip, dtype=np.uint8).flatten()
    offset = model.tex_adr[tex_id]
    model.tex_data[offset: offset + pixels.size] = pixels
    mujoco.mjr_uploadTexture(model, context, tex_id)

def skybox_texture_id(model: mujoco.MjModel) -> int | None:
    skybox_type = mujoco.mjtTexture.mjTEXTURE_SKYBOX
    for tid in range(model.ntex):
        if model.tex_type[tid] == skybox_type:
            print(f"TEX_ID: {tid} (skybox)")
            return tid
    return None

def randomise_mesh_scale_in_spec(spec: mujoco.MjSpec, prefixes: Sequence[str]):
    scale = np.random.uniform(0.9, 1.1, size=3)
    print(f"\n[Scale] ×{np.round(scale,3)}")
    for mesh in spec.meshes:
        if any(mesh.name.startswith(p) for p in prefixes):
            if mesh.scale is None:
                mesh.scale = np.ones(3)
            mesh.scale = scale
            print(f"  {mesh.name}: {np.round(mesh.scale, 3)}")

def randomise_vine_positions(model: mujoco.MjModel, spec: mujoco.MjSpec, vine_ids: np.ndarray):
    lo = np.array([0.2, -0.1, 0.6])
    hi = np.array([0.4,  0.1, 0.8])
    new_pos = np.random.uniform(lo, hi, (len(vine_ids), 3))
    model.body_pos[vine_ids, :] = new_pos
    for bid, pos in zip(vine_ids, new_pos):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        print(f"  {name}: {np.round(pos, 3)}")
        for body_spec in spec.worldbody.bodies:
            if body_spec.name == name:
                body_spec.pos = pos.tolist()
                break

def move_unique_assets(dest: mujoco.MjSpec, src: mujoco.MjSpec):
    for coll in ("meshes", "textures", "materials", "skins", "hfields"):
        dlist, slist = getattr(dest, coll), getattr(src, coll)
        have = {a.name for a in dlist if a.name}
        for a in list(slist):
            if a.name and a.name not in have:
                dlist.append(a)
                slist.remove(a)

# ----------------------- MODIFIED: random per-vine template -----------------------
def build_scene(n_vines: int = 8, choices: Sequence[str] = XML_CHOICES):
    """Build a scene with n_vines, picking a random XML per vine from *choices*."""
    scene_spec = mujoco.MjSpec.from_file("scene.xml")
    roots: list[str] = []

    seen_asset_sources: set[str] = set()
    for i in range(n_vines):
        pick = random.choice(choices)
        vine_spec = mujoco.MjSpec.from_file(pick)

        # Merge assets for each template the first time it appears
        if pick not in seen_asset_sources:
            move_unique_assets(scene_spec, vine_spec)
            seen_asset_sources.add(pick)

        frame = scene_spec.worldbody.add_frame(name=f"vine_mount_{i}")
        root = frame.attach_body(vine_spec.worldbody.bodies[0], suffix=f"_{i}")
        roots.append(root.name)
        print(f"[build_scene] Placed vine {i}: {pick} -> root '{root.name}'")
    return scene_spec, roots
# -------------------------------------------------------------------------------

def main():
    sky_pool = find_skybox_images()
    if not sky_pool:
        print(f"No skybox images found in {SKYBOX_DIR}; B key will do nothing.")
    else:
        print(f"Loaded {len(sky_pool)} skybox textures from {SKYBOX_DIR}")

    # Determine which vine templates are available
    available_choices = [p for p in XML_CHOICES if Path(p).exists()]
    if not available_choices:
        raise FileNotFoundError("None of strawb.xml / leaves.xml / strawb_fork.xml exist.")
    if len(available_choices) < len(XML_CHOICES):
        missing = set(XML_CHOICES) - set(available_choices)
        print(f"Warning: missing vine XMLs: {', '.join(missing)}")
    print(f"Sampling vines from: {available_choices}")

    spec, root_names = build_scene(8, choices=available_choices)
    model = spec.compile()
    data = mujoco.MjData(model)

    tex_id = skybox_texture_id(model)
    if tex_id is None:
        print("Warning: model has no skybox texture; B key disabled.")

    vine_ids = np.array([model.body(n).id for n in root_names])
    strawberry_prefixes = ["leaf", "strawberry", "strawberry_leaves", "strawberry_collision"]

    renderer = mujoco.Renderer(model, 480, 480)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.lookat[:] = (0.3, 0.0, 0.7)
    cam.distance  = 0.5
    cam.elevation = 0
    cam.azimuth   = 90

    proc = psutil.Process()
    print("\nControls:  S = scale   |   P = position   |   B = skybox   |   Q / Esc = quit\n")

    try:
        while True:
            # cam.azimuth += 0.2
            t0 = time.time()
            mujoco.mj_step(model, data)

            renderer.update_scene(data, camera=cam)
            bgr = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)

            mem = proc.memory_info().rss / (1024*1024)
            cv2.putText(bgr, f"Memory: {mem:6.1f} MB", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)

            cv2.imshow("MuJoCo Simulation", bgr)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

            elif key == ord("s"):
                randomise_mesh_scale_in_spec(spec, strawberry_prefixes)
                model, data = spec.recompile(model, data)
                old_model, old_data = model, data
                renderer.close()
                del renderer, old_model, old_data
                gc.collect()
                renderer = mujoco.Renderer(model, 480, 480)
                tex_id = skybox_texture_id(model)

            elif key == ord("p"):
                randomise_vine_positions(model, spec, vine_ids)
                mujoco.mj_forward(model, data)

            elif key == ord("b") and sky_pool and tex_id is not None:
                texture = random.choice(sky_pool)
                apply_texture(model, renderer._mjr_context, tex_id, texture)

            dt = model.opt.timestep - (time.time() - t0)
            if dt > 0:
                time.sleep(dt)

    finally:
        renderer.close()
        cv2.destroyAllWindows()
        print("Exited cleanly.")


if __name__ == "__main__":
    if not Path("scene.xml").exists():
        raise FileNotFoundError("scene.xml")
    # Don’t force all three; we handle missing ones in main()
    main()
