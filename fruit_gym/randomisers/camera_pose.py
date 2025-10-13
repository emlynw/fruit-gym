# camera_pose.py
from __future__ import annotations
from typing import Sequence, Iterable, Optional
import numpy as np
import mujoco

from .base import Randomiser


def _quat_mul(q2, q1):
    # MuJoCo quat: [w, x, y, z]
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w2*w1 - x2*x1 - y2*y1 - z2*z1,
            w2*x1 + x2*w1 + y2*z1 - z2*y1,
            w2*y1 - x2*z1 + y2*w1 + z2*x1,
            w2*z1 + x2*y1 - y2*x1 + z2*w1,
        ],
        dtype=np.float32,
    )


def _axis_angle_to_quat(axis, angle):
    s = np.sin(angle * 0.5)
    return np.array([np.cos(angle * 0.5), *(s * axis)], dtype=np.float32)


class CameraPoseRandomiser(Randomiser):
    """
    Adds noise to camera poses (model.cam_pos / model.cam_quat).

    Parameters
    ----------
    cam_names : list of camera names to affect. None → all cameras.
    pos_lo / pos_hi : length-3 bounds added to cam_pos (translation jitter).
    rot_enabled     : if False, no rotation noise is applied.
    ang_range       : (min,max) radians for axis-angle magnitude.
    yaw_only        : if True, rotate only about global Z.
    """

    affects_spec = False
    needs_ctx = False

    def __init__(
        self,
        cam_names: Optional[Iterable[str]] = None,
        pos_lo: Sequence[float] = (0.0, 0.0, 0.0),
        pos_hi: Sequence[float] = (0.0, 0.0, 0.0),
        rot_enabled: bool = False,
        ang_range: tuple[float, float] = (-0.15, 0.15),
        yaw_only: bool = False,
    ):
        self.cam_names = None if cam_names is None else [str(n) for n in cam_names]
        self.pos_lo = np.asarray(pos_lo, float)
        self.pos_hi = np.asarray(pos_hi, float)
        self.ang_lo, self.ang_hi = ang_range
        self.rot_enabled = rot_enabled
        self.yaw_only = yaw_only

    # ------------------------------------------------------------------ #

    def _target_cam_ids(self, model: mujoco.MjModel):
        if self.cam_names is None:
            return list(range(model.ncam))
        ids = []
        for name in self.cam_names:
            cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
            if cid != -1:
                ids.append(cid)
        return ids

    def apply(self, *, spec, model, data, rng, ctx=None):
        cam_ids = self._target_cam_ids(model)
        if not cam_ids:
            return

        for cid in cam_ids:
            # --- translation jitter ---
            dpos = rng.uniform(self.pos_lo, self.pos_hi)
            model.cam_pos[cid] = (model.cam_pos[cid] + dpos).astype(np.float32)

            # --- rotation jitter (optional) ---
            if not self.rot_enabled:
                continue

            axis = np.array([0.0, 0.0, 1.0], float) if self.yaw_only else rng.normal(size=3)
            axis /= (np.linalg.norm(axis) + 1e-12)
            angle = rng.uniform(self.ang_lo, self.ang_hi)
            dq = _axis_angle_to_quat(axis, angle)

            q = model.cam_quat[cid]
            q = q / (np.linalg.norm(q) + 1e-12)
            q_new = _quat_mul(dq, q)
            model.cam_quat[cid] = (q_new / (np.linalg.norm(q_new) + 1e-12)).astype(np.float32)
