# randomisers/table_simple.py
from __future__ import annotations
import numpy as np
import mujoco
from .base import Randomiser

class TableRandomiser(Randomiser):
    """
    Model-pass: jitter table position and randomise its RGBA.
    If p_absent > 0, sometimes hide it *and* disable collisions.

    Parameters
    ----------
    body_name : MuJoCo body name that holds the table geom (default: 'table')
    geom_name : Geom name to recolor/hide (default: 'table')
    pos_lo/pos_hi : 3D jitter bounds ADDED to the baseline body_pos
    rgb_lo/rgb_hi : per-channel RGB ranges (0..1)
    alpha        : alpha when present
    p_absent     : probability the table is fully absent (alpha=0 & no collisions)
    """

    affects_spec = False
    needs_ctx = False

    def __init__(
        self,
        *,
        body_name: str = "table",
        geom_name: str = "table",
        pos_lo = (-0.02, -0.05, -0.02),
        pos_hi = ( 0.02,  0.05,  0.02),
        rgb_lo = (0.2, 0.2, 0.2),
        rgb_hi = (0.9, 0.9, 0.9),
        alpha: float = 1.0,
        p_absent: float = 0.0,
    ):
        self.body_name = str(body_name)
        self.geom_name = str(geom_name)
        self.pos_lo = np.asarray(pos_lo, dtype=float)
        self.pos_hi = np.asarray(pos_hi, dtype=float)
        self.rgb_lo = np.asarray(rgb_lo, dtype=float)
        self.rgb_hi = np.asarray(rgb_hi, dtype=float)
        self.alpha = float(alpha)
        self.p_absent = float(p_absent)

        # cached on first use
        self._cached = False
        self._bid = None
        self._gid = None
        self._base_body_pos = None

    def _ensure_cached(self, model: mujoco.MjModel):
        if self._cached:
            return
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.body_name)
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, self.geom_name)
        if bid == -1 or gid == -1:
            raise ValueError(f"TableSimpleRandomiser: body '{self.body_name}' or geom '{self.geom_name}' not found.")
        self._bid = int(bid)
        self._gid = int(gid)
        self._base_body_pos = model.body_pos[self._bid].copy()
        self._cached = True

    def apply(self, *, spec, model, data, rng, ctx=None):
        self._ensure_cached(model)

        # --- position jitter around baseline (no cumulative drift) ---
        dpos = rng.uniform(self.pos_lo, self.pos_hi)
        model.body_pos[self._bid] = (self._base_body_pos + dpos).astype(np.float32)

        # --- absent toggle ---
        if rng.random() < self.p_absent:
            rgba = model.geom_rgba[self._gid].copy()
            rgba[:3] = 0.0  # color doesn't matter, it's invisible
            rgba[3] = 0.0
            model.geom_rgba[self._gid] = rgba.astype(np.float32)
            model.geom_contype[self._gid] = 0
            model.geom_conaffinity[self._gid] = 0
            return

        # --- present: random RGBA + enable collisions ---
        rgb = rng.uniform(self.rgb_lo, self.rgb_hi)
        rgb = np.clip(rgb, 0.0, 1.0)
        rgba = np.array([rgb[0], rgb[1], rgb[2], self.alpha], dtype=np.float32)
        model.geom_rgba[self._gid] = rgba
        # simple default: ensure it collides
        model.geom_contype[self._gid] = 1
        model.geom_conaffinity[self._gid] = 1
