"""
optimized_scene.py  •  MuJoCo 3.3.x

Keys
----
S   randomise strawberry mesh scale (spec → recompile, leak‑free)
P   randomise vine root positions   (mjModel in‑place, instant)
B   randomise skybox texture        (in‑place texture upload, instant)
Q / Esc   quit
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
import gc                 # <-- new import


from PIL import Image
import cv2


SKYBOX_DIR = Path.cwd() / "textures" / "skyboxes"
SKYBOX_EXTS = {".png", ".jpg", ".jpeg"}

def find_skybox_images() -> list[Path]:
    if not SKYBOX_DIR.exists():
        return []
    return [p for p in SKYBOX_DIR.iterdir()
            if p.suffix.lower() in SKYBOX_EXTS and p.is_file()]


# (row, col) of each face in the 3×4 cross
_CROSS_POS = {
    "U": (0, 1),
    "L": (1, 0),
    "F": (1, 1),
    "R": (1, 2),
    "B": (1, 3),
    "D": (2, 1),
}

# order of faces in MuJoCo’s 1×6 strip
_STRIP_ORDER = ["F", "B", "U", "D", "R", "L"] 
# _STRIP_ORDER = ["U", "L", "F", "R", "B", "D"]  # OpenGL order: U, L, R, F, B, D


def _prepare_face(face_img: Image.Image, face: str) -> Image.Image:
    """Replicate MuJoCo’s internal transforms for each cube face."""

    # 1. Flip Y: Pillow’s origin is top‑left, OpenGL’s is bottom‑left
    face_img = face_img.transpose(Image.FLIP_TOP_BOTTOM)

    # 2. Rotate so every tile’s +X matches the Front face +X
    if face == "U":
        face_img = face_img.rotate(-90, expand=True)   # 90° CCW
    elif face == "D":
        face_img = face_img.rotate(90,  expand=True)   # 90° CW
    else:
        face_img = face_img.rotate(180,  expand=True)
    return face_img


def _cross_to_strip(img: Image.Image, out_w: int, out_h: int) -> Image.Image:
    """Convert .U..LFRB.D.. cross → LRFBUD *vertical* strip (out_w × out_h)."""
    tile_in  = img.height // 3      # square tiles in the source
    tile_out = out_w                # size of one tile in the strip (512)
    strip = Image.new("RGB", (out_w, out_h))

    for i, face in enumerate(_STRIP_ORDER):
        r, c = _CROSS_POS[face]
        crop = img.crop((c * tile_in, r * tile_in,
                         (c + 1) * tile_in, (r + 1) * tile_in))
        crop = _prepare_face(crop, face)

        if tile_in != tile_out:
            crop = crop.resize((tile_out, tile_out), Image.LANCZOS)

        # paste DOWN the single column, not across
        strip.paste(crop, (0, i * tile_out))

    return strip


def apply_texture(model: mujoco.MjModel,
                  context: mujoco.MjrContext,
                  tex_id: int,
                  path: Path):
    """Replace texture *tex_id* pixels with the image at *path* (RGB8)."""
    try:
        img = Image.open(path).convert("RGB")
        print(f"Applying texture: {path.name}")
    except FileNotFoundError:
        print(f"Warning: texture not found: {path}")
        return

    out_h, out_w = model.tex_height[tex_id], model.tex_width[tex_id]
    print(f"image size: {img.size}, target size: {out_w}x{out_h}")

    # fast path – already the exact LRFBUD strip
    if img.size == (out_w, out_h):
        strip = img.transpose(Image.FLIP_TOP_BOTTOM)  # only flip Y
    else:
        # assume 3×4 cross and convert
        if img.height % 3 or img.width % 4:
            print(f"{path.name} is neither 1×6 nor 3×4; skipped.")
            return
        strip = _cross_to_strip(img, out_w, out_h)

    # upload to MuJoCo / OpenGL
    pixels = np.asarray(strip, dtype=np.uint8).flatten()
    offset = model.tex_adr[tex_id]
    model.tex_data[offset: offset + pixels.size] = pixels
    mujoco.mjr_uploadTexture(model, context, tex_id)

# --------------------------------------------------------------------------- #
#  Texture helper                                                              #
# --------------------------------------------------------------------------- #


def skybox_texture_id(model: mujoco.MjModel) -> int | None:
    """Return the texture id of the first skybox texture (or None)."""
    skybox_type = mujoco.mjtTexture.mjTEXTURE_SKYBOX
    for tid in range(model.ntex):
        if model.tex_type[tid] == skybox_type:
            print(f"TEX_ID: {tid} (skybox)")
            return tid
    return None


# --------------------------------------------------------------------------- #
#  Other helpers (unchanged)                                                   #
# --------------------------------------------------------------------------- #
def randomise_mesh_scale_in_spec(spec: mujoco.MjSpec, prefixes: Sequence[str]):
    scale = np.random.uniform(0.9, 1.1, size=3)
    print(f"\n[Scale] ×{scale}")
    for mesh in spec.meshes:
        if any(mesh.name.startswith(p) for p in prefixes):
            if mesh.scale is None:
                mesh.scale = np.ones(3)
            mesh.scale = scale
            print(f"  {mesh.name}: {np.round(mesh.scale, 3)}")


def randomise_vine_positions(model: mujoco.MjModel, spec: mujoco.MjSpec, vine_ids: np.ndarray):
    lo = np.array([0.00, -1.0, 0.4])
    hi = np.array([1.00,  1.0, 0.8])
    new_pos = np.random.uniform(lo, hi, (len(vine_ids), 3))
    model.body_pos[vine_ids, :] = new_pos
    for bid, pos in zip(vine_ids, new_pos):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        print(f"  {name}: {np.round(pos, 3)}")
        # Update the spec as well
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


def build_scene(n_vines: int = 8):
    scene_spec = mujoco.MjSpec.from_file("scene.xml")
    roots = []
    first = True
    for i in range(n_vines):
        vine_spec = mujoco.MjSpec.from_file("strawb.xml")
        if first:
            move_unique_assets(scene_spec, vine_spec)
            first = False
        frame = scene_spec.worldbody.add_frame(name=f"vine_mount_{i}")
        root = frame.attach_body(vine_spec.worldbody.bodies[0], suffix=f"_{i}")
        roots.append(root.name)
    return scene_spec, roots

# --------------------------------------------------------------------------- #
#  Main                                                                        #
# --------------------------------------------------------------------------- #
def main():
    # ---------- skybox candidates -------------------------------------------
    sky_pool = find_skybox_images()
    if not sky_pool:
        print(f"No skybox images found in {SKYBOX_DIR}; B key will do nothing.")
    else:
        print(f"Loaded {len(sky_pool)} skybox textures from {SKYBOX_DIR}")


    # ---------- build & compile --------------------------------------------
    spec, root_names = build_scene(8)
    model, data = spec.compile(), None
    data = mujoco.MjData(model)

    tex_id = skybox_texture_id(model)
    if tex_id is None:
        print("Warning: model has no skybox texture; B key disabled.")

    vine_ids = np.array([model.body(n).id for n in root_names])
    strawberry_prefixes = ["strawberry", "strawberry_leaves", "strawberry_collision"]

    renderer = mujoco.Renderer(model, 480, 480)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.lookat[:] = (0.0, 0.0, 0.6)
    cam.distance  = 3.0
    cam.elevation = 0
    cam.azimuth   = 0

    proc = psutil.Process()
    print("\nControls:  S = scale   |   P = position   |   B = skybox   |   Q / Esc = quit\n")

    try:
        i = 0 
        while True:
            cam.azimuth += 0.2
            # cam.elevation = 1000 * np.sin(i)
            # cam.elevation += 0.2
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
                tex_id = skybox_texture_id(model)  # id may change after compile

            elif key == ord("p"):
                randomise_vine_positions(model, spec, vine_ids)
                mujoco.mj_forward(model, data)

            elif key == ord("b") and sky_pool and tex_id is not None:
                texture = random.choice(sky_pool)
                apply_texture(model, renderer._mjr_context,
                              tex_id, texture)
                grid_raw  = spec.textures[tex_id].gridlayout   # whatever the binding returns

            # real‑time pacing
            dt = model.opt.timestep - (time.time() - t0)
            if dt > 0:
                time.sleep(dt)

    finally:
        renderer.close()
        cv2.destroyAllWindows()
        print("Exited cleanly.")


if __name__ == "__main__":
    for f in ("scene.xml", "strawb.xml"):
        if not Path(f).exists():
            raise FileNotFoundError(f)
    main()
