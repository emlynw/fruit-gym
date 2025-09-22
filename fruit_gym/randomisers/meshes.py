from .base import Randomiser
from typing import Sequence
import mujoco
import numpy as np

def _iter_spec_nodes(node):
    yield node
    for b in getattr(node, "bodies", []):
        yield from _iter_spec_nodes(b)
    for f in getattr(node, "frames", []):
        yield from _iter_spec_nodes(f)

class MeshVariantRandomiser(Randomiser):
    """
    Swap the mesh used by each strawberry visual geom to a random variant.

    • affects_spec=True → triggers recompile
    • Groups by numeric suffix in the geom's name so a single berry stays consistent.
    """

    affects_spec = True
    needs_ctx = False

    def __init__(
        self,
        geom_prefixes: Sequence[str] = ("block_visual", "block1_visual"),
        mesh_pool: Sequence[str] | None = None,   # explicit list of mesh names
        mesh_name_prefix: str = "strawberry_",    # used if mesh_pool is None
    ):
        self.geom_prefixes = tuple(geom_prefixes)
        self.mesh_pool = tuple(mesh_pool) if mesh_pool else None
        self.mesh_name_prefix = mesh_name_prefix

    @staticmethod
    def _instance_id(name: str) -> str | None:
        if "_" not in name:
            return None
        tail = name.rsplit("_", 1)[1]
        return tail if tail.isdigit() else None

    def _looks_like_target(self, name: str) -> bool:
        return bool(name) and name.startswith(self.geom_prefixes)

    def _discover_mesh_pool(self, spec: mujoco.MjSpec) -> list[str]:
        pool = [m.name for m in spec.meshes if m.name and m.name.startswith(self.mesh_name_prefix)]
        pool = sorted(set(pool))
        print(f"[MeshVariant] discovered {len(pool)} meshes with prefix '{self.mesh_name_prefix}': {pool[:8]}{' …' if len(pool)>8 else ''}")
        return pool
    
    def set_geom_mesh(geom, new_mesh_name: str):
        # works across mujoco versions
        if hasattr(geom, "mesh"):
            geom.mesh = new_mesh_name
        elif hasattr(geom, "meshname"):
            geom.meshname = new_mesh_name
        else:
            print(f"[MeshVariant] geom '{getattr(geom,'name','?')}' has no mesh field")

    def apply(self, *, spec, model, data, rng, ctx=None):
        if spec is None:
            raise ValueError("MeshVariantRandomiser needs a spec when affects_spec=True")

        # Build candidate mesh list
        if self.mesh_pool:
            mesh_names = list(self.mesh_pool)
            print(f"[MeshVariant] using explicit mesh_pool: {mesh_names}")
        else:
            mesh_names = self._discover_mesh_pool(spec)

        if not mesh_names:
            print("[MeshVariant] No candidate meshes found. "
                  "Set mesh_pool=['strawberry_1', ...] or mesh_name_prefix='strawberry_' to match your assets.")
            return

        name_to_mesh = {m.name: m for m in spec.meshes if m.name}
        missing = [n for n in mesh_names if n not in name_to_mesh]
        if missing:
            print(f"[MeshVariant] Skipping missing meshes: {missing}")
            mesh_names = [n for n in mesh_names if n in name_to_mesh]
            if not mesh_names:
                print("[MeshVariant] No valid meshes remain; aborting.")
                return

        id2mesh: dict[str | None, str] = {}
        n_changed = 0
        n_targets = 0

        for node in _iter_spec_nodes(spec.worldbody):
            for g in getattr(node, "geoms", []):
                gname = getattr(g, "name", "") or ""
                if not self._looks_like_target(gname):
                    continue
                # only retarget mesh-type visuals
                if getattr(g, "meshname", None) is None:
                    print(f"[MeshVariant] Skipping non-mesh geom '{gname}'")
                    continue

                n_targets += 1
                inst = self._instance_id(gname)
                if inst not in id2mesh:
                    id2mesh[inst] = rng.choice(mesh_names)

                chosen_mesh = id2mesh[inst]
                if getattr(g, "mesh", None) != chosen_mesh:
                    g.meshname = chosen_mesh
                    n_changed += 1

        print(f"[MeshVariant] Targets: {n_targets} geoms, Instances: {len(id2mesh)}, Reassigned: {n_changed}")
