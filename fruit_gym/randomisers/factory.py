"""fruit_gym.randomisers.factory

Build the list of concrete **Randomiser** objects for a given YAML
configuration.  The YAML *must* follow the latest structure described in
`configs/example.yaml` (see README) – notably each sub‑section lives under
`domain_randomization:` and contains an `enabled: bool` flag.

Only the randomisers that already exist in the code‑base are wired up here.
If a section is `enabled: true` but we don’t have a corresponding
randomiser implementation yet, we simply ignore it and print a friendly
message (so the env still works).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Mapping, Tuple

import numpy as np

from fruit_gym.randomisers import (
    Randomiser,
    SkyboxRandomiser,
    AssignBerryMaterialsRandomiser,
    LightingRandomiser,
    PoseRandomiser,
    ScaleRandomiser,
    SpawnerRandomiser,
    RobotPoseRandomiser,
)

__all__ = ["build_randomisers"]

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _get_range(pair) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a YAML `[lo, hi]` list (or a scalar) into two *np.float32* arrays.

    If *pair* is a single scalar (e.g. `0.1`) we turn it into
    `(-pair, +pair)` which is handy for colour jitter ranges.
    """
    if isinstance(pair, (int, float)):
        lo, hi = -pair, +pair
    else:
        lo, hi = pair
    return np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)

# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------

def build_randomisers(cfg: Mapping, *, xml_dir: Path | str = "") -> List[Randomiser]:
    """Create the *episode* randomisers for the current YAML configuration.

    Parameters
    ----------
    cfg
        The full YAML dict loaded via ``yaml.safe_load``.
    xml_dir
        Unused for now – kept for API compatibility.
    """
    rand: list[Randomiser] = []
    dr = cfg.get("domain_randomization", {})

    # ---------------- skybox ------------------------------------------------
    if dr.get("skybox", {}).get("enabled", False):
        sky_dir = Path(xml_dir) / "textures" / "skyboxes"
        rand.append(SkyboxRandomiser(sky_dir))

    # ---------------- lighting ---------------------------------------------
    if dr.get("lighting", {}).get("enabled", False):
        l_cfg = dr["lighting"]
        # each *_range entry has structure {low: [...], high: [...]} in YAML
        def _lo_hi(sub_key, default):
            sub = l_cfg.get(sub_key, {})
            return sub.get("low", default), sub.get("high", default)

        rand.append(
            LightingRandomiser(
                pos_range_low=np.asarray(l_cfg.get("position_range_low", (-.8, -.5, -.05)), float),
                pos_range_high=np.asarray(l_cfg.get("position_range_high", (1.2, .5, .2)), float),
                diffuse_range=_lo_hi("diffuse_range", 0.05),
                ambient_range=_lo_hi("ambient_range", 0.0),
                specular_range=_lo_hi("specular_range", 0.0),
            )
        )

    # ---------------- spawner ----------------------------------------------
    if dr.get("objects_count", {}).get("enabled", False):
        s_cfg = dr["objects_count"]

        raw_choices = s_cfg.get("xml_choices")
        if raw_choices is None:
            raw_choices = ["strawb_stiff.xml", "strawb_fork.xml", "leaves.xml"]

        xml_choices = [Path(xml_dir) / name for name in raw_choices]

        rand.append(
            SpawnerRandomiser(
                xml_choices=xml_choices,
                min_count=s_cfg.get("min_fruits", 4),
                max_count=s_cfg.get("max_fruits", 8),
                mount_prefix=s_cfg.get("mount_prefix", "vine_"),
            )
        )

    # --------------- object positions --------------------------------------
    if dr.get("object_positions", {}).get("enabled", False):
        o_cfg = dr["object_positions"]
        pos_lo = np.asarray(o_cfg.get("position_range_low", (0, 0, 0)), float)
        pos_hi = np.asarray(o_cfg.get("position_range_high", (0, 0, 0)), float)
        rot_enabled = o_cfg.get("rotation_enabled", True)
        rot_angle_range = tuple(o_cfg.get("rotation_angle_range", (-1.57, 1.57)))
        yaw_only = o_cfg.get("yaw_only", True)
        rand.append(PoseRandomiser(name_prefix="vine", pos_lo=pos_lo, pos_hi=pos_hi, rot_enabled=rot_enabled, rot_angle_range=rot_angle_range, yaw_only=yaw_only))

    # ---------------- object scale -----------------------------------------
    if dr.get("object_scale", {}).get("enabled", False):
        s_cfg = dr["object_scale"]
        scale_range = tuple(s_cfg.get("scale_range", [0.8, 1.2]))
        rand.append(
            ScaleRandomiser(
                prefixes=["strawberry", "strawberry_leaves"],
                scale_range=scale_range,
            )
        )

    # ---------------- strawb texture ------------------------------------------------
    if dr.get("strawberry_texture", {}).get("enabled", False):
        rand.append(
            AssignBerryMaterialsRandomiser(
                material_names=["berry_mat_red", "berry_mat_green", "berry_mat_mix"]
            )
        )

    # ---------------- robot pose -------------------------------------------
    if dr.get("robot_pose", {}).get("enabled", True):
        r_cfg = dr["robot_pose"]
        pos_lo = np.asarray(r_cfg.get("position_range_low", (-0.04, -0.05, 0.00)), float)
        pos_hi = np.asarray(r_cfg.get("position_range_high", (0.04, 0.05, 0.10)), float)
        rot_enabled = r_cfg.get("rotation_enabled", True)
        ang_range = tuple(r_cfg.get("angle_range", (-0.15, 0.15)))
        yaw_only = r_cfg.get("yaw_only", False)
        rand.append(
            RobotPoseRandomiser(
                pos_lo=pos_lo,
                pos_hi=pos_hi,
                rot_enabled=rot_enabled,
                yaw_only=yaw_only,
                ang_range=ang_range,
            )
        )

    return rand
