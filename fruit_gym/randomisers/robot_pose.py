# robot_pose.py
from __future__ import annotations
from typing import Sequence
import numpy as np
import mujoco

from .base import Randomiser


def _quat_mul(q2, q1):
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


def _axis_angle_to_quat(axis, angle):
    s = np.sin(angle / 2.0)
    return np.array([np.cos(angle / 2.0), *(s * axis)], dtype=np.float32)


class RobotPoseRandomiser(Randomiser):
    """Adds noise to the initial TCP pose.

    Parameters
    ----------
    pos_lo / pos_hi : 3-element bounds for translation noise.
    ang_range       : (min,max) radians for axis-angle magnitude.
    yaw_only        : if True, rotate only around global Z.
    pos_only        : if True, **disable orientation noise entirely**.
    """

    affects_spec = False
    needs_ctx = False

    def __init__(
        self,
        pos_lo: Sequence[float] = (-0.04, -0.05, 0.00),
        pos_hi: Sequence[float] = ( 0.04,  0.05, 0.10),
        rot_enabled: bool = False,
        ang_range: tuple[float, float] = (-0.15, 0.15),
        yaw_only: bool = False,
    ):
        self.pos_lo = np.asarray(pos_lo, float)
        self.pos_hi = np.asarray(pos_hi, float)
        self.ang_lo, self.ang_hi = ang_range
        self.rot_enabled = rot_enabled
        self.yaw_only = yaw_only

    # ------------------------------------------------------------------ #

    def apply(self, *, spec, model, data, rng, ctx=None):
        # --- position ---------------------------------------------------
        dpos = rng.uniform(self.pos_lo, self.pos_hi)
        data.mocap_pos[0] += dpos

        # --- orientation -----------------------------------------------
        if not self.rot_enabled:
            return  # skip rotation noise altogether

        axis = (
            np.array([0, 0, 1], float)
            if self.yaw_only
            else rng.normal(size=3)
        )
        axis /= np.linalg.norm(axis)
        angle = rng.uniform(self.ang_lo, self.ang_hi)
        dq = _axis_angle_to_quat(axis, angle)
        data.mocap_quat[0] = _quat_mul(dq, data.mocap_quat[0])
        print(f"RobotPoseRandomiser: pos={dpos}, rot={dq}, yaw_only={self.yaw_only}, angle={angle:.2f} rad")
