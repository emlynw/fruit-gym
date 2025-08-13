from __future__ import annotations
from typing import Sequence
import numpy as np
import mujoco
from .base import Randomiser


# ---------------- quaternion helpers (same as before) ---------------------- #
def _quat_mul(q2: np.ndarray, q1: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w2 * w1 - x2 * x1 - y2 * y1 - z2 * z1,
            w2 * x1 + x2 * w1 + y2 * z1 - z2 * y1,
            w2 * y1 - x2 * z1 + y2 * w1 + z2 * x1,
            w2 * z1 + x2 * y1 - y2 * x1 + z2 * w1,
        ],
        dtype=np.float32,
    )


def _random_axis_angle(
    rng: np.random.Generator,
    angle_lo: float,
    angle_hi: float,
    yaw_only: bool,
) -> tuple[np.ndarray, float]:
    angle = rng.uniform(angle_lo, angle_hi)
    if yaw_only:
        axis = np.array([0.0, 0.0, 1.0])
    else:
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
    return axis, angle


def _axis_angle_to_quat(axis: np.ndarray, angle: float) -> np.ndarray:
    s = np.sin(angle / 2.0)
    return np.array([np.cos(angle / 2.0), *(s * axis)], dtype=np.float32)


# ---------------- pose randomiser ----------------------------------------- #
class PoseRandomiser(Randomiser):
    """
    Jitters every body whose name starts with `name_prefix`.

    Parameters
    ----------
    name_prefix      Filter for body names (`startswith` test).
    pos_lo, pos_hi   AABB for **absolute** positions.
    rot_angle_range  `(min,max)` radians for rotation magnitude.
    yaw_only         True → rotate about world-Z only.
    rot_enabled      False → **skip rotation noise** entirely.
    """

    affects_spec = False
    needs_ctx = False

    def __init__(
        self,
        name_prefix: str = "vine",
        pos_lo: Sequence[float] = (0.0, -1.0, 0.4),
        pos_hi: Sequence[float] = (1.0,  1.0, 0.8),
        rot_enabled: bool = True,
        rot_angle_range: tuple[float, float] = (-0.15, 0.15),
        yaw_only: bool = False,
        leaves_tag: str = "leaves",
        leaves_z_offset: float = 0.1
    ):
        self.prefix = str(name_prefix)
        self.pos_lo = np.asarray(pos_lo, dtype=float)
        self.pos_hi = np.asarray(pos_hi, dtype=float)
        self.rot_enabled = rot_enabled
        self.rot_lo, self.rot_hi = rot_angle_range
        self.yaw_only = yaw_only
        self.leaves_tag = leaves_tag
        self.leaves_z_offset = float(leaves_z_offset)

    # --------------------------------------------------------------------- #

    def apply(self, *, spec, model, data, rng, ctx=None):
        for bid in range(model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
            if not name.startswith(self.prefix):
                continue

            # ----- position -------------------------------------------------
            model.body_pos[bid] += rng.uniform(self.pos_lo, self.pos_hi)

            # NEW: small fixed lift for leaves
            if self.leaves_tag and self.leaves_tag in name:
                model.body_pos[bid][2] += self.leaves_z_offset

            # ----- orientation (optional) ----------------------------------
            if self.rot_enabled:
                axis, angle = _random_axis_angle(
                    rng, self.rot_lo, self.rot_hi, self.yaw_only
                )
                q_delta = _axis_angle_to_quat(axis, angle)
                model.body_quat[bid] = _quat_mul(q_delta, model.body_quat[bid])
