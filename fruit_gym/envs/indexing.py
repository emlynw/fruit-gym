from dataclasses import dataclass, field
from typing import Dict, List, Optional
import mujoco
from .mujoco_utils import geom_name, mat_name_for_geom, body_parent

def instance_key(name: str) -> Optional[str]:
    if not name:
        return None
    parts = name.split("_")
    if len(parts) >= 3 and parts[-1].isdigit() and parts[-2].isdigit():
        return f"{parts[-2]}_{parts[-1]}"
    if parts[-1].isdigit():
        return parts[-1]
    return None

@dataclass
class FruitInstance:
    fruit_geoms: List[int] = field(default_factory=list)
    calyx_geoms: List[int] = field(default_factory=list)
    stem_geom: Optional[int] = None
    fruit_body: Optional[int] = None
    ripe: bool = False
    material: str = ""

def is_ripe_material(mat_name: str, ripe_mats: set[str]) -> bool:
    return mat_name in ripe_mats

def index_fruits_and_stems(model: mujoco.MjModel, *, ripe_mats: set[str]):
    fruit_candidates, calyx_candidates, stem_candidates = [], [], []

    for gid in range(int(model.ngeom)):
        name = geom_name(model, gid)
        if name.startswith("fruit_"):
            fruit_candidates.append(gid)
        elif name.startswith("calyx_"):
            calyx_candidates.append(gid)
        elif name.startswith("stem"):
            stem_candidates.append(gid)

    geom_bodyid = model.geom_bodyid

    def nearest_stem_for_body(start_bid: int) -> Optional[int]:
        seen = set()
        bid = int(start_bid)
        while bid not in seen and bid >= 0:
            seen.add(bid)
            g0 = int(model.body_geomadr[bid])
            n  = int(model.body_geomnum[bid])
            for k in range(n):
                gid = g0 + k
                if geom_name(model, gid).startswith("stem"):
                    return gid
            bid = body_parent(model, bid)
        return None

    fruit_instances: Dict[str, FruitInstance] = {}

    for gid in fruit_candidates:
        k = instance_key(geom_name(model, gid))
        if k is None:
            continue
        d = fruit_instances.setdefault(k, FruitInstance())
        d.fruit_geoms.append(gid)
        if d.fruit_body is None:
            d.fruit_body = int(geom_bodyid[gid])

    for gid in calyx_candidates:
        k = instance_key(geom_name(model, gid))
        if k is None or k not in fruit_instances:
            continue
        fruit_instances[k].calyx_geoms.append(gid)

    for k, d in fruit_instances.items():
        if d.fruit_geoms:
            any_fruit_gid = d.fruit_geoms[0]
            d.stem_geom = nearest_stem_for_body(int(geom_bodyid[any_fruit_gid]))

        mat = mat_name_for_geom(model, d.fruit_geoms[0]) if d.fruit_geoms else ""
        d.material = mat
        d.ripe = is_ripe_material(mat, ripe_mats)

    ripe_ids   = sorted([k for k, d in fruit_instances.items() if d.ripe])
    unripe_ids = sorted([k for k, d in fruit_instances.items() if not d.ripe])
    stem_to_fruit = {int(d.stem_geom): k for k, d in fruit_instances.items() if d.stem_geom is not None}

    return fruit_instances, ripe_ids, unripe_ids, stem_to_fruit
