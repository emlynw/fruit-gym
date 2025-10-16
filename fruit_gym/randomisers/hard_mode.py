# fruit_gym/randomisers/hard_mode.py
from __future__ import annotations
from typing import Iterable, Optional
import mujoco
import numpy as np
import re

from .base import Randomiser


class HardMode(Randomiser):
    """
    With probability p, set 'hard mode' for this episode:
      - Group berry visuals into instances by trailing numeric suffix in the geom name.
      - Compute each instance's mean world X-position from data.geom_xpos.
      - Mark the 'farther +X' half as RIPE and the others as UNRIPE, by assigning materials.

    Config (YAML under domain_randomization.hard_mode):
      enabled: true
      p: 0.5
      name_match: "block"               # which geoms are berry visuals
      ripe_materials: ["r1","r2","r3"]  # must exist in the compiled model
      unripe_materials: ["g1","g2","g3"]

    Notes:
      - Does nothing when p < rng.random() (i.e., easy mode this episode).
      - Runs in-place (no spec changes). Place AFTER any generic assignment and AFTER
        EnsureMinRipeBerries if you want hard-mode to win.
    """

    affects_spec = False
    needs_ctx = False

    def __init__(
        self,
        *,
        p: float = 0.5,
        name_match: str = "block",
        ripe_materials: Iterable[str] = ("r1", "r2", "r3"),
        unripe_materials: Iterable[str] = ("g1", "g2", "g3"),
        **_ignored, 
    ):
        self.p = float(p)
        self.name_match = str(name_match).lower()
        self.ripe_materials = tuple(ripe_materials)
        self.unripe_materials = tuple(unripe_materials)

    # ---- helpers ----
    @staticmethod
    def _instance_id(name: str) -> Optional[str]:
        # Use trailing numeric suffix as instance id (e.g., "block_2" -> "2")
        if not name or "_" not in name:
            return None
        tail = name.rsplit("_", 1)[1]
        return tail if tail.isdigit() else None

    def _looks_like_target(self, name: str) -> bool:
        return self.name_match in (name or "").lower()

    def _material_ids(self, model: mujoco.MjModel, names: Iterable[str]) -> list[int]:
        out = []
        for nm in names:
            try:
                out.append(int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, nm)))
            except mujoco.Error:
                pass
        return out

    # ---- main ----
    def apply(self, *, spec, model, data, rng, ctx=None):
        # coin flip for hard mode
        if self.p <= 0.0 or rng.random() >= self.p:
            return
        
        mujoco.mj_forward(model, data)

        # resolve material ids
        ripe_ids = self._material_ids(model, self.ripe_materials)
        unripe_ids = self._material_ids(model, self.unripe_materials)
        if not ripe_ids or not unripe_ids:
            return

        ripe_set = set(ripe_ids)

        # collect geoms per *fruit instance* (by trailing numeric suffix)
        inst2gids: dict[str, list[int]] = {}
        for gid in range(int(model.ngeom)):
            gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
            if not self._looks_like_target(gname):
                continue
            inst = self._instance_id(gname)
            if inst is None:
                continue
            inst2gids.setdefault(inst, []).append(gid)
        if not inst2gids:
            return

        # 1) how many instances are currently ripe? (before hard-mode)
        currently_ripe_insts = []
        for inst, gids in inst2gids.items():
            if any(int(model.geom_matid[g]) in ripe_set for g in gids):
                currently_ripe_insts.append(inst)
        n_ripe_target = len(currently_ripe_insts)  # preserve this count

        if n_ripe_target == 0:
            # nothing to do; respect upstream pipeline's decision
            return

        # 2) compute mean world X for each instance
        inst_x = []
        for inst, gids in inst2gids.items():
            xs = [float(data.geom_xpos[g][0]) for g in gids]
            inst_x.append((inst, float(np.mean(xs)) if xs else -np.inf))

        # 3) sort by +X and assign: top N -> ripe, rest -> unripe
        inst_x.sort(key=lambda t: t[1])           # ascending
        top = {inst for inst, _ in inst_x[-n_ripe_target:]}  # top-N by +X

        ripe_ids_arr = np.asarray(ripe_ids, dtype=int)
        unripe_ids_arr = np.asarray(unripe_ids, dtype=int)

        for inst, gids in inst2gids.items():
            mat_pool = ripe_ids_arr if inst in top else unripe_ids_arr
            chosen = int(rng.choice(mat_pool))
            for gid in gids:
                model.geom_matid[gid] = chosen
