from .base import Randomiser
import numpy as np
from typing import Sequence

class ScaleRandomiser(Randomiser):
    """Uniformly scale meshes whose names start with any of *prefixes*."""

    affects_spec = True  # mutates the *spec* → env must recompile
    needs_ctx = False  # does not need a MjrContext

    def __init__(self, prefixes: Sequence[str],
                 scale_range: tuple[float, float] = (0.5, 2.0)):
        self.prefixes = tuple(prefixes)
        self.scale_lo, self.scale_hi = scale_range

    def _instance_id(self, name: str) -> str | None:
        """
        Extract the trailing numeric tag (e.g.  'strawberry_leaves_7' → '7').
        Returns *None* if no numeric suffix is found, in which case that mesh
        is scaled independently.
        """
        if "_" not in name:
            return None
        tail = name.rsplit("_", 1)[1]
        return tail if tail.isdigit() else None

    def apply(self, *, spec, model, data, rng, ctx=None):
        if spec is None:
            raise ValueError("ScaleRandomiser needs a spec when affects_spec=True")
        id2factor = {}
        for mesh in spec.meshes:
            if mesh.name and mesh.name.startswith(self.prefixes):
                inst = self._instance_id(mesh.name)
                if inst not in id2factor:
                    id2factor[inst] = rng.uniform(self.scale_lo, self.scale_hi)

                factor = id2factor[inst]
                mesh.scale = (np.ones(3) if mesh.scale is None else mesh.scale) * factor
                print(f"[Scale] mesh '{mesh.name}' (id '{inst}') scaled by {factor:.3f}")
