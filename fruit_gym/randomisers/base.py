from abc import ABC, abstractmethod
import mujoco
from typing import Protocol, Sequence, Optional
import numpy as np

class Randomiser(ABC):
    """Base class for all episode randomisers."""

    #: Whether the randomiser mutates the *spec* and therefore requires the env
    #: to call `spec.recompile()` afterwards.
    affects_spec: bool = False

    @abstractmethod
    def apply(self, *,
                spec: Optional[mujoco.MjSpec],
                model: mujoco.MjModel,
                data: mujoco.MjData,
                rng: np.random.Generator) -> None: ...