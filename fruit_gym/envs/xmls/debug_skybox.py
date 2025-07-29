"""
optimized_scene.py  •  MuJoCo 3.3.x

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


SKYBOX_DIR = Path.cwd() / "textures" / "skyboxes"
SKYBOX_EXTS = {".png", ".jpg", ".jpeg"}

def find_skybox_images() -> list[Path]:
    if not SKYBOX_DIR.exists():
        return []
    return [p for p in SKYBOX_DIR.iterdir()
            if p.suffix.lower() in SKYBOX_EXTS and p.is_file()]

# --------------------------------------------------------------------------- #
#  Texture helper with exact mapping                                          #
# --------------------------------------------------------------------------- #
def debug_grid_mapping():
    """Debug the exact grid mapping for MuJoCo."""
    gridlayout = ".U..LFRB.D.."
    gridsize = (3, 4)  # 3 columns, 4 rows
    
    print("MuJoCo Grid Layout Analysis:")
    print(f"gridlayout: '{gridlayout}'")
    print(f"gridsize: {gridsize} (3 columns, 4 rows)")
    print()
    
    # Map the 12 characters to grid positions
    print("Linear index -> Grid position (col, row) -> Face:")
    for i, face in enumerate(gridlayout):
        col = i % 3  # column = index mod 3
        row = i // 3  # row = index div 3
        print(f"  {i:2d} -> ({col}, {row}) -> '{face}'")
    
    print()
    print("Grid visualization:")
    for row in range(4):
        row_str = ""
        for col in range(3):
            idx = row * 3 + col
            face = gridlayout[idx] if idx < len(gridlayout) else "?"
            row_str += f"| {face} "
        row_str += "|"
        print(row_str)
    print()

def convert_with_exact_mapping(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Convert using exact MuJoCo grid mapping - simple approach."""
    
    # Source: 4 cols × 3 rows (standard cubemap cross)
    source_cols, source_rows = 4, 3
    source_face_w = img.width // source_cols
    source_face_h = img.height // source_rows
    
    # Target: 3 cols × 4 rows (MuJoCo format)  
    target_cols, target_rows = 3, 4
    target_face_w = target_w // target_cols  # 512÷3 = 170
    target_face_h = target_h // target_rows  # 3072÷4 = 768
    
    print(f"Source face size: {source_face_w} × {source_face_h}")
    print(f"Target face size: {target_face_w} × {target_face_h}")
    print(f"Target texture size: {target_w} × {target_h}")
    
    # Extract faces from standard 4×3 cross layout
    faces = {}
    
    # Standard cross layout positions:
    # Row 0: [empty, Up, empty, empty]
    faces['U'] = img.crop((source_face_w, 0, 2*source_face_w, source_face_h))
    
    # Row 1: [Left, Front, Right, Back]  
    faces['L'] = img.crop((0, source_face_h, source_face_w, 2*source_face_h))
    faces['F'] = img.crop((source_face_w, source_face_h, 2*source_face_w, 2*source_face_h))
    faces['R'] = img.crop((2*source_face_w, source_face_h, 3*source_face_w, 2*source_face_h))
    faces['B'] = img.crop((3*source_face_w, source_face_h, 4*source_face_w, 2*source_face_h))
    
    # Row 2: [empty, Down, empty, empty]
    faces['D'] = img.crop((source_face_w, 2*source_face_h, 2*source_face_w, 3*source_face_h))
    
    # Resize all faces to target dimensions (MuJoCo expects these exact dimensions)
    for key in faces:
        faces[key] = faces[key].resize((target_face_w, target_face_h), Image.LANCZOS)
    
    # Create target image
    result = Image.new('RGB', (target_w, target_h), (0, 0, 0))
    
    # Map to MuJoCo positions according to ".U..LFRB.D.."
    gridlayout = ".U..LFRB.D.."
    
    for i, face_char in enumerate(gridlayout):
        if face_char != '.' and face_char in faces:
            col = i % 3
            row = i // 3
            x = col * target_face_w
            y = row * target_face_h
            
            print(f"Placing face '{face_char}' at grid ({col}, {row}) -> pixel ({x}, {y})")
            result.paste(faces[face_char], (x, y))
    
    return result

def apply_texture_exact_mapping(model: mujoco.MjModel,
                               context: mujoco.MjrContext,
                               tex_id: int,
                               path: Path):
    """Apply texture with exact grid mapping."""
    try:
        img = Image.open(path)
        print(f"Applying texture: {path.name}")
        print(f"Loaded image: {img.size}, mode: {img.mode}")
    except FileNotFoundError:
        print(f"Warning: texture not found: {path}")
        return

    h, w = model.tex_height[tex_id], model.tex_width[tex_id]
    print(f"Target dimensions: {w} × {h}")
    
    # Convert to RGB
    if img.mode != "RGB":
        print(f"Converting from {img.mode} to RGB")
        img = img.convert("RGB")
    
    # Debug the grid mapping first (only once)
    if not hasattr(apply_texture_exact_mapping, 'debug_shown'):
        debug_grid_mapping()
        apply_texture_exact_mapping.debug_shown = True
    
    # Convert format if needed
    if img.size != (w, h):
        source_aspect = img.width / img.height
        
        if abs(source_aspect - 4/3) < 0.1:
            print("Converting 4:3 cubemap to MuJoCo format with exact mapping")
            img = convert_with_exact_mapping(img, w, h)
        else:
            print("Non-standard format, simple resize")
            img = img.resize((w, h), Image.LANCZOS)
    
    # Apply texture
    pixel_bytes = img.tobytes('raw', 'RGB')
    pixels_flat = np.frombuffer(pixel_bytes, dtype=np.uint8)
    
    offset = model.tex_adr[tex_id]
    model.tex_data[offset : offset + pixels_flat.size] = pixels_flat
    mujoco.mjr_uploadTexture(model, context, tex_id)
    print("Texture applied successfully")
    print("-" * 50)

def skybox_texture_id(model: mujoco.MjModel) -> int | None:
    """Return the texture id of the first skybox texture (or None)."""
    skybox_type = mujoco.mjtTexture.mjTEXTURE_SKYBOX
    for tid in range(model.ntex):
        if model.tex_type[tid] == skybox_type:
            return tid
    return None

# --------------------------------------------------------------------------- #
#  Other helpers (unchanged)                                                   #
# --------------------------------------------------------------------------- #
def randomise_mesh_scale_in_spec(spec: mujoco.MjSpec, prefixes: Sequence[str]):
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

    vine_ids = np.array([model.body(n).id for n in root_names])
    strawberry_prefixes = ["strawberry", "strawberry_leaves", "strawberry_collision"]

    renderer = mujoco.Renderer(model, 480, 480)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.lookat[:] = (0.0, 0.0, 0.6)
    cam.distance  = 1.0
    cam.elevation = 0
    cam.azimuth   = 0

    tex_id = skybox_texture_id(model)
    if tex_id is None:
        print("Warning: model has no skybox texture; B key disabled.")

    proc = psutil.Process()
    print("\nControls:  S = scale   |   P = position   |   B = skybox   |   Q / Esc = quit\n")

    try:
        while True:
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
                renderer.close(); renderer = mujoco.Renderer(model, 480, 480)
                tex_id = skybox_texture_id(model)  # id may change after compile

            elif key == ord("p"):
                randomise_vine_positions(model, vine_ids)
                mujoco.mj_forward(model, data)

            elif key == ord("b") and sky_pool and tex_id is not None:
                texture = random.choice(sky_pool)
                apply_texture_exact_mapping(model, renderer._mjr_context,
                                           tex_id, texture)

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