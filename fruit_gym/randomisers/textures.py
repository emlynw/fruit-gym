from .base import Randomiser
from pathlib import Path
from PIL import Image
import mujoco
import numpy as np
from typing import Optional

SKYBOX_EXTS: set[str] = {".png", ".jpg", ".jpeg"}

# (row, col) of each face in the 3×4 cross
_CROSS_POS = {
    "U": (0, 1),
    "L": (1, 0),
    "F": (1, 1),
    "R": (1, 2),
    "B": (1, 3),
    "D": (2, 1),
}

# order of faces in MuJoCo’s 1×6 strip (LRFBUD)
_STRIP_ORDER = ["F", "B", "U", "D", "R", "L"]

def find_skybox_images(root: Path) -> list[Path]:
    """Return all image files in *root* that have an accepted extension."""
    if not root.exists():
        return []
    return [p for p in root.iterdir() if p.suffix.lower() in SKYBOX_EXTS and p.is_file()]


def _prepare_face(face_img: Image.Image, face: str) -> Image.Image:
    """Replicate MuJoCo’s internal transforms for each cube face."""
    face_img = face_img.transpose(Image.FLIP_TOP_BOTTOM)  # Flip Y axis
    if face == "U":
        face_img = face_img.rotate(-90, expand=True)
    elif face == "D":
        face_img = face_img.rotate(90, expand=True)
    else:
        face_img = face_img.rotate(180, expand=True)
    return face_img


def _cross_to_strip(img: Image.Image, out_w: int, out_h: int) -> Image.Image:
    """Convert 3×4 cross → vertical LRFBUD strip of size (*out_w*, *out_h*)."""
    tile_in = img.height // 3
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


def skybox_texture_id(model: mujoco.MjModel) -> Optional[int]:
    """Return the ID of the *first* skybox texture, or *None* if not present."""
    skybox_type = mujoco.mjtTexture.mjTEXTURE_SKYBOX
    for tid in range(model.ntex):
        if model.tex_type[tid] == skybox_type:
            return tid
    return None


def upload_skybox_texture(model: mujoco.MjModel, context: mujoco.MjrContext, *,
                           tex_id: int, path: Path) -> None:
    """Replace texture *tex_id* in‑place with pixels from *path*.  PNG/JPG only."""
    try:
        img = Image.open(path).convert("RGB")
    except FileNotFoundError:
        print(f"[Skybox] File not found: {path}")
        return

    out_h, out_w = model.tex_height[tex_id], model.tex_width[tex_id]

    if img.size == (out_w, out_h):
        strip = img.transpose(Image.FLIP_TOP_BOTTOM)
    else:
        if img.height % 3 or img.width % 4:
            print(f"[Skybox] {path.name} not 3×4 cross nor 1×6 strip – skipped.")
            return
        strip = _cross_to_strip(img, out_w, out_h)

    pixels = np.asarray(strip, dtype=np.uint8).flatten()
    offset = model.tex_adr[tex_id]
    model.tex_data[offset: offset + pixels.size] = pixels
    mujoco.mjr_uploadTexture(model, context, tex_id)


class SkyboxRandomiser(Randomiser):
    """Swap the skybox texture with a random file from *skybox_dir*."""

    affects_spec = False  # in‑place OpenGL upload
    needs_ctx = True  # requires a MjrContext for texture upload
    
    def __init__(self, skybox_dir: Path):
        self._skyboxes = find_skybox_images(skybox_dir)
        if not self._skyboxes:
            print(f"[Skybox] No images found in {skybox_dir}; randomiser disabled.")

    def apply(self, *, spec, model, data, rng, ctx=None):
        if not self._skyboxes:
            return
        tex_id = skybox_texture_id(model)
        if tex_id is None:
            return
        path = rng.choice(self._skyboxes)
        upload_skybox_texture(model, ctx, tex_id=tex_id, path=path)

class AssignBerryMaterialsRandomiser(Randomiser):
    """
    After compile, assign each strawberry visual geom a random material from
    ['berry_mat_red', 'berry_mat_green', 'berry_mat_mix'] that you define in scene.xml.
    """
    affects_spec = False
    needs_ctx = False

    def __init__(self, material_names=None, name_match="block_visual"):
        self.material_names = material_names or ["berry_mat_red", "berry_mat_green", "berry_mat_mix"]
        self.name_match = name_match.lower()

    @staticmethod
    def _looks_like_block_visual(name: str) -> bool:
        n = (name or "").lower()
        return ("block_visual" in n) or (n.startswith("block") and "visual" in n)

    def apply(self, *, spec, model, data, rng, ctx=None):
        # Resolve material IDs that actually exist in the compiled model
        mat_ids = []
        for name in self.material_names:
            try:
                mid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, name)
                mat_ids.append(int(mid))
            except mujoco.Error:
                pass
        if not mat_ids:
            print(f"[BerryAssign] None of {self.material_names} found in model; skipping.")
            return

        mat_ids = np.asarray(mat_ids, dtype=int)
        n = 0
        for gid in range(int(model.ngeom)):
            gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
            if self._looks_like_block_visual(gname):
                model.geom_matid[gid] = int(rng.choice(mat_ids))
                n += 1
        print(f"[BerryAssign] Rebound {n} geoms to random berry materials.")