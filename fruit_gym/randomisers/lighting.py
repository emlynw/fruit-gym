from .base import Randomiser
from pathlib import Path
import mujoco
import numpy as np
from typing import Sequence, Optional


class LightingRandomiser(Randomiser):
    """Randomise a single movable light plus the global headlight."""

    affects_spec = False

    def __init__(self,
                 pos_range_low: Sequence[float] = (-0.8, -0.5, -0.05),
                 pos_range_high: Sequence[float] = (1.2, 0.5, 0.2),
                 diffuse_range: Sequence[float] = (0.05, 0.3),
                 ambient_range: Sequence[float] = (0.0, 0.2),
                 specular_range: Sequence[float] = (0.0, 0.5)):
        self.pos_low = np.asarray(pos_range_low, dtype=float)
        self.pos_high = np.asarray(pos_range_high, dtype=float)
        self.diffuse_rng = diffuse_range
        self.ambient_rng = ambient_range
        self.specular_rng = specular_range

    def apply(self, *, spec, model, data, rng):
        # move light0 (assumed to exist)
        light_bid = model.body("light0").id if "light0" in model._body_name2id else None
        if light_bid is not None:
            model.body_pos[light_bid] = rng.uniform(self.pos_low, self.pos_high)

        # colour/intensity
        def _rand_rgb(lo_hi):
            lo, hi = lo_hi
            return rng.uniform(lo, hi, size=3)

        model.light_diffuse[0] = _rand_rgb(self.diffuse_rng)
        model.light_ambient[0] = _rand_rgb(self.ambient_rng)
        model.light_specular[0] = _rand_rgb(self.specular_rng)

        # headlight tweaks (very subtle)
        jitter = rng.uniform(-0.1, 0.1, size=9)
        model.vis.headlight.diffuse += jitter[:3]
        model.vis.headlight.ambient += jitter[3:6]
        model.vis.headlight.specular += jitter[6:9]