"""fruit_gym.envs.barebones_env

A *minimal* MuJoCo‑based Gymnasium environment whose only responsibility
is to compile an XML once, call a list of **Randomiser** objects at
reset, and expose `reset()` / `step()` so you can very quickly test each
randomiser in isolation.

It deliberately omits:
    • reward shaping
    • robot controllers
    • fancy observation dictionaries
    • grasp logic, etc.

Usage example
-------------

```python
from pathlib import Path
from fruit_gym.randomisers import (
    SkyboxRandomiser, LightingRandomiser,
    VinePoseRandomiser, StrawberryScaleRandomiser,
)
from fruit_gym.envs.barebones_env import RandomiserTestEnv

sky_dir = Path.cwd() / "textures" / "skyboxes"
rand_list = [
    SkyboxRandomiser(sky_dir),
    LightingRandomiser(),
    VinePoseRandomiser(["vine_mount_0", "vine_mount_1"]),
    StrawberryScaleRandomiser(["strawberry"]),
]

env = RandomiserTestEnv(xml_path="scene.xml", randomisers=rand_list)
obs = env.reset()       # randomisers executed here
for _ in range(240):    # simulate 1 s @ 240 Hz
    env.step(env.action_space.sample())
    env.render()
```
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Sequence

import mujoco
import numpy as np
from gymnasium.envs.mujoco import MujocoEnv

from fruit_gym.randomisers import Randomiser

__all__ = ["RandomiserTestEnv"]


class RandomiserTestEnv(MujocoEnv):
    """Super‑thin wrapper around MuJoCo + list[Randomiser]."""

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self,
                 xml_path: str | Path,
                 randomisers: Sequence[Randomiser],
                 width: int = 480,
                 height: int = 480,
                 render_mode: str = "human"):
        self._base_spec = mujoco.MjSpec.from_file(str(xml_path))
        self._randomisers = list(randomisers)
        self._rng = np.random.default_rng()

        # compile once – will recompile on reset if needed
        self.spec = copy.deepcopy(self._base_spec)
        model, data = self.spec.compile(), None

        super().__init__(model, frame_skip=1, render_mode=render_mode,
                         width=width, height=height, camera_id=0)

        # trivial continuous action‑space: one noop scalar
        from gymnasium.spaces import Box
        self.action_space = Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    # ------------------------------------------------------------------ reset
    def reset_model(self):  # called by MujocoEnv.reset()
        self.spec = copy.deepcopy(self._base_spec)
        needs_recompile = False
        for r in self._randomisers:
            r.apply(spec=self.spec if r.affects_spec else None,
                    model=self.model, data=self.data, rng=self._rng)
            needs_recompile |= r.affects_spec

        if needs_recompile:
            self.model, self.data = self.spec.recompile(self.model, self.data)

        mujoco.mj_forward(self.model, self.data)
        return {}

    # ------------------------------------------------------------------- step
    def step(self, action):
        mujoco.mj_step(self.model, self.data)
        obs = {}
        reward = 0.0  # no task → always zero
        terminated = False
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info
