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
    After compile, assign each target geom a random material from `material_names`.

    name_match can be:
      - str: substring must appear in the geom name
      - list/tuple[str]: ANY of the substrings may match
      - dict: advanced matching:
          {"all": ["block", "visual"]}        # all substrings must appear
          {"any": ["block_visual", "block1_visual"]}
          {"prefix": ["block_visual", "block1_visual"]}
          {"suffix": ["_visual", "_display"]}
          {"regex": r"^(block(_visual\d+)?|berry_visual)$"}
      - callable: fn(geom_name: str) -> bool
    """

    affects_spec = False
    needs_ctx = False

    def __init__(self, material_names=None, name_match="block_visual"):
        self.material_names = material_names or ["berry_mat_red", "berry_mat_green", "berry_mat_mix"]
        self._test = self._make_name_tester(name_match)

    # ---------- matching helpers ----------
    @staticmethod
    def _make_name_tester(name_match):
        # callable provided
        if callable(name_match):
            return lambda n: bool(name_match(n or ""))

        # regex
        if isinstance(name_match, dict) and "regex" in name_match:
            import re
            rx = re.compile(name_match["regex"])
            return lambda n: bool(rx.search((n or "").lower()))

        # dict forms
        if isinstance(name_match, dict):
            def tester(n):
                s = (n or "").lower()
                if "all" in name_match and not all(sub.lower() in s for sub in name_match["all"]):
                    return False
                if "any" in name_match and not any(sub.lower() in s for sub in name_match["any"]):
                    return False
                if "prefix" in name_match:
                    prefs = tuple(p.lower() for p in name_match["prefix"])
                    if not s.startswith(prefs):
                        return False
                if "suffix" in name_match:
                    sufs = tuple(x.lower() for x in name_match["suffix"])
                    if not s.endswith(sufs):
                        return False
                return True
            return tester

        # list/tuple → any substring
        if isinstance(name_match, (list, tuple)):
            subs = tuple(x.lower() for x in name_match)
            return lambda n: any(sub in (n or "").lower() for sub in subs)

        # str → single substring
        sub = str(name_match).lower()
        return lambda n: sub in (n or "").lower()

    # ---------- main ----------
    def apply(self, *, spec, model, data, rng, ctx=None):
        # resolve material IDs present in the compiled model
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
            if self._test(gname):
                model.geom_matid[gid] = int(rng.choice(mat_ids))
                n += 1
        print(f"[BerryAssign] Rebound {n} geoms to random berry materials.")


class EnsureMinRipeBerries(Randomiser):
    """
    Model-pass: ensure at least `min_ripe` fruit instances use a 'ripe' material.

    - Matches geoms by name (default: names containing 'block').
    - Groups by trailing numeric suffix so both halves of the berry (block_1_N, block_2_N)
      get updated together to the same ripe material (one of r1, r2, r3 by default).
    """
    affects_spec = False
    needs_ctx = False

    def __init__(self,
                 ripe_materials = ("r1", "r2", "r3"),
                 name_match = "block",         # which geoms count as berry visuals
                 min_ripe: int = 2):
        self.ripe_materials = tuple(ripe_materials)
        self.name_match = str(name_match).lower()
        self.min_ripe = int(min_ripe)

    @staticmethod
    def _instance_id(name: str) -> str | None:
        if "_" not in (name or ""):
            return None
        tail = name.rsplit("_", 1)[1]
        return tail if tail.isdigit() else None

    def _looks_like_target(self, name: str) -> bool:
        return self.name_match in (name or "").lower()

    def apply(self, *, spec, model, data, rng, ctx=None):
        # Resolve ripe material IDs that actually exist
        ripe_mat_ids = []
        for nm in self.ripe_materials:
            try:
                ripe_mat_ids.append(int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, nm)))
            except mujoco.Error:
                pass
        if not ripe_mat_ids:
            print(f"[MinRipe] No ripe materials found among {self.ripe_materials}; skipping.")
            return

        # Gather target geoms per fruit instance
        inst2gids: dict[str | None, list[int]] = {}
        for gid in range(int(model.ngeom)):
            gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
            if not self._looks_like_target(gname):
                continue
            inst = self._instance_id(gname)
            inst2gids.setdefault(inst, []).append(gid)

        if not inst2gids:
            print("[MinRipe] No target geoms found; skipping.")
            return

        ripe_set = set(ripe_mat_ids)
        # Which instances are already ripe?
        ripe_insts = []
        non_ripe_insts = []
        for inst, gids in inst2gids.items():
            # Consider the instance ripe if ANY of its geoms is ripe
            if any(int(model.geom_matid[g]) in ripe_set for g in gids):
                ripe_insts.append(inst)
            else:
                non_ripe_insts.append(inst)

        need = max(0, self.min_ripe - len(ripe_insts))
        if need == 0:
            print(f"[MinRipe] Already have {len(ripe_insts)} ripe instances (>= {self.min_ripe}); noop.")
            return

        # Promote some non-ripe instances to ripe
        if need > len(non_ripe_insts):
            need = len(non_ripe_insts)  # best effort
        to_promote = list(rng.choice(non_ripe_insts, size=need, replace=False))

        changed = 0
        for inst in to_promote:
            gids = inst2gids.get(inst, [])
            chosen_mat = int(rng.choice(ripe_mat_ids))
            for gid in gids:
                model.geom_matid[gid] = chosen_mat
                changed += 1

        print(f"[MinRipe] Promoted {len(to_promote)} instances to ripe; reassigned {changed} geoms.")