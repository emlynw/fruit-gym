from abc import ABC, abstractmethod
import mujoco
from typing import Optional
import numpy as np

class Randomiser(ABC):
    """Base class for all episode randomisers."""

    #: Whether the randomiser mutates the *spec* and therefore requires the env
    #: to call `spec.recompile()` afterwards.
    affects_spec: bool = False
    needs_ctx: bool = False  # whether the randomiser needs a MjrContext

    @abstractmethod
    def apply(self, *,
                spec: Optional[mujoco.MjSpec],
                model: mujoco.MjModel,
                data: mujoco.MjData,
                rng: np.random.Generator,
                ctx:  Optional[mujoco.MjrContext] = None, 
                ) -> None: ...