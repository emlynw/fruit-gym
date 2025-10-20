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
    EnsureMinRipeBerries,
    VineColorRandomiser,
    LightingRandomiser,
    PoseRandomiser,
    ScaleRandomiser,
    MeshVariantRandomiser,
    TableRandomiser,
    SpawnerRandomiser,
    RobotPoseRandomiser,
    CameraPoseRandomiser,
    HardMode,
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
        
        fruit_xmls = s_cfg.get("fruit_xmls")
        if fruit_xmls is None:
            fruit_xmls = ["strawb_stiff.xml"]
        fruit_xmls = [Path(xml_dir) / name for name in fruit_xmls]

        rand.append(
            SpawnerRandomiser(
                xml_choices=fruit_xmls,
                min_count=s_cfg.get("min_fruits", 4),
                max_count=s_cfg.get("max_fruits", 8),
                mount_prefix=s_cfg.get("mount_prefix", "vine_"),
                ensure_min_fruit=s_cfg.get("max_ripe", 2),  # ensure at least this many ripe berries
            )
        )

        leaf_xmls = s_cfg.get("leaf_xmls")
        if leaf_xmls is None:
            leaf_xmls = ["leaves.xml"]
        leaf_xmls = [Path(xml_dir) / name for name in leaf_xmls]
        rand.append(
            SpawnerRandomiser(
                xml_choices=leaf_xmls,
                min_count=s_cfg.get("min_leaves", 2),
                max_count=s_cfg.get("max_leaves", 6),
                mount_prefix="leaves_",
                ensure_min_fruit=0,  # ensure at least this many ripe berries
            )
        )

    # ---------------- mesh variants ----------------------------------------
    if dr.get("strawberry_mesh", {}).get("enabled", False):
        mv_cfg = dr["strawberry_mesh"]
        rand.append(
            MeshVariantRandomiser(
                geom_prefixes=tuple(mv_cfg.get("geom_prefixes", ["fruit"])),
                mesh_pool=mv_cfg.get("mesh_pool"),  # e.g. ["strawberry_1","strawberry_2","strawberry_3"]
                mesh_name_prefix=mv_cfg.get("mesh_name_prefix", "strawberry_"),
            )
        )

    if dr.get("calyx_mesh", {}).get("enabled", False):
        cm_cfg = dr["calyx_mesh"]
        rand.append(
            MeshVariantRandomiser(
                geom_prefixes=tuple(cm_cfg.get("geom_prefixes", ["calyx"])),  # target geoms named calyx1, calyx2, ...
                mesh_pool=cm_cfg.get("mesh_pool"),                             # e.g., ["calyx1","calyx2","calyx3"]
                mesh_name_prefix=cm_cfg.get("mesh_name_prefix", "calyx"),      # or auto-discover by prefix
            )
        )

    if dr.get("leaf_mesh", {}).get("enabled", False):
        lm_cfg = dr["leaf_mesh"]
        rand.append(
            MeshVariantRandomiser(
                geom_prefixes=tuple(lm_cfg.get("geom_prefixes", ["leaf"])),  # target geoms named leaf1, leaf2, ...
                mesh_pool=lm_cfg.get("mesh_pool"),                             # e.g., ["leaf1","leaf2","leaf3"]
                mesh_name_prefix=lm_cfg.get("mesh_name_prefix", "leaf"),      # or auto-discover by prefix
            )
        )


    # --------------- object positions --------------------------------------
    if dr.get("object_positions", {}).get("enabled", False):
        o_cfg = dr["object_positions"]
        pos_lo = np.asarray(o_cfg.get("position_range_low", (0, 0, 0)), float)
        pos_hi = np.asarray(o_cfg.get("position_range_high", (0, 0, 0)), float)
        dev_std = o_cfg.get("position_deviation_std", (0.06, 0.06, 0.06))
        rot_enabled = o_cfg.get("rotation_enabled", True)
        rot_angle_range = tuple(o_cfg.get("rotation_angle_range", (-1.57, 1.57)))
        cluster_rot_angle_range = tuple(o_cfg.get("cluster_rotation_angle_range", (-0.05, 0.05)))
        yaw_only = o_cfg.get("yaw_only", True)
        rand.append(PoseRandomiser(name_prefix="vine", pos_lo=pos_lo, pos_hi=pos_hi, dev_std=dev_std, rot_enabled=rot_enabled, 
                                   rot_angle_range=rot_angle_range, cluster_rot_angle_range=cluster_rot_angle_range, yaw_only=yaw_only))

    # ---------------- object scale -----------------------------------------
    if dr.get("object_scale", {}).get("enabled", False):
        s_cfg = dr["object_scale"]
        fruit_scale_range = tuple(s_cfg.get("fruit_scale_range", [0.6, 1.4]))
        rand.append(
            ScaleRandomiser(
                prefixes=["sb", "calyx"],
                scale_range=fruit_scale_range,
            )
        )
        # Scale leaves too
        leaf_scale_range = tuple(s_cfg.get("leaf_scale_range", [0.3, 0.5]))
        rand.append(
            ScaleRandomiser(
                prefixes=["leaf"],
                scale_range=leaf_scale_range,
            )
        )

    # ---------------- strawb texture ------------------------------------------------
    if dr.get("strawberry_texture", {}).get("enabled", False):
        rand.append(
            AssignBerryMaterialsRandomiser(
                # material_names=["berry_mat_red", "berry_mat_green", "berry_mat_mix"]
                material_names=["r1", "r2", "r3", "g1", "g2", "g3", "u1", "u2", "u3"],
                name_match="fruit"
            )
        )

        # Ensure at least two 'ripe' berries exist
        rand.append(
            EnsureMinRipeBerries(
                ripe_materials=("r1","r2","r3"),
                name_match="fruit",
                min_ripe=2,
            )
        )

    if dr.get("strawberry_texture", {}).get("enabled", False):
        rand.append(
            AssignBerryMaterialsRandomiser(
                # material_names=["berry_mat_red", "berry_mat_green", "berry_mat_mix"]
                material_names=["calyx"],
                name_match="calyx"
            )
        )

    if dr.get("leaf_texture", {}).get("enabled", False):
        rand.append(
            AssignBerryMaterialsRandomiser(
                material_names=["leaf_g1", "leaf_g2", "leaf_s1", "leaf_s2", "leaf_s3"],
                name_match="leaf"
            )
        )

     # ---------------- vine color (tint stems/vines green) -------------------
    if dr.get("vine_color", {}).get("enabled", False):
        vc = dr["vine_color"]
        rand.append(
            VineColorRandomiser(
                base_rgba=tuple(vc.get("base_rgba", (0.18, 0.30, 0.12, 1.0))),
                jitter=tuple(vc.get("jitter", (0.1, 0.1, 0.1))),
                ensure_green_margin=float(vc.get("ensure_green_margin", 0.02)),
                alpha=float(vc.get("alpha", 1.0)),
                prefixes=tuple(vc.get("prefixes", ("seg", "stem"))),
                regex=vc.get("regex", None),
                per_instance=bool(vc.get("per_instance", True)),
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

    # ---------------- camera pose ------------------------------------------
    if dr.get("camera_pose", {}).get("enabled", False): 
        c_cfg = dr["camera_pose"]
        pos_lo = np.asarray(c_cfg.get("position_range_low", (0.0, 0.0, 0.0)), float)
        pos_hi = np.asarray(c_cfg.get("position_range_high", (0.0, 0.0, 0.0)), float)
        rot_enabled = c_cfg.get("rotation_enabled", False)
        ang_range = tuple(c_cfg.get("angle_range", (-0.15, 0.15)))
        yaw_only = c_cfg.get("yaw_only", False)
        cam_names = c_cfg.get("camera_names", None)  # None → all cameras
        rand.append(
            CameraPoseRandomiser(
                cam_names=cam_names,
                pos_lo=pos_lo,
                pos_hi=pos_hi,
                rot_enabled=rot_enabled,
                yaw_only=yaw_only,
                ang_range=ang_range,
            )
        )

    # ---------------- table -------------------------------------------------
    if dr.get("table", {}).get("enabled", False):
        t_cfg = dr["table"]
        pos_lo = np.asarray(t_cfg.get("position_range_low", (-0.05, -0.02, -0.05)), float)
        pos_hi = np.asarray(t_cfg.get("position_range_high", (0.05, 0.02, 0.05)), float)
        rand.append(
            TableRandomiser(
                body_name=t_cfg.get("body_name", "table"),
                geom_name=t_cfg.get("geom_name", "table"),
                pos_lo=pos_lo,
                pos_hi=pos_hi,
                rgb_lo=t_cfg.get("rgb_lo", (0.3, 0.3, 0.3)),
                rgb_hi=t_cfg.get("rgb_hi", (0.9, 0.9, 0.9)),
                alpha=float(t_cfg.get("alpha", 1.0)),
                p_absent=float(t_cfg.get("p_absent", 0.3)),
            )
        )

    # -------------------- HARD MODE (ripe = farther +X) --------------------
    if dr.get("hard_mode", {}).get("enabled", False):
        rand.append(HardMode(**dr["hard_mode"]))

    return rand
