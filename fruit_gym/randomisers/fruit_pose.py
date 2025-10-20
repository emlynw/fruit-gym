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


# ---------------- pose randomiser (clustered) ------------------------------ #
class PoseRandomiser(Randomiser):
    """
    Clustered pose jitter: bodies whose name starts with `name_prefix` are moved
    around a single per-episode **base pose** (shared cluster center), plus a
    **per-body deviation** around that base.

    Parameters
    ----------
    name_prefix
        Filter for body names (`startswith` test).
    pos_lo, pos_hi
        AABB for the **base translation** (episode-wise shared shift).
        Sampled once per apply(), then added to each matching body.
    dev_std
        Per-body zero-mean deviation std (Normal noise). Scalar or 3-vector.
        Small -> tight cluster; large -> spread out.
    rot_enabled
        If True, apply rotations. First apply an optional small **base rotation**
        shared by all bodies, then a per-body extra rotation (see ranges below).
    cluster_rot_angle_range
        Angle range (radians) for the **base rotation** (shared).
    rot_angle_range
        Angle range (radians) for **per-body** rotation noise.
    yaw_only
        True → rotate about world-Z only (for both base & per-body rotations).
    leaves_tag
        If substring is present in a body name, apply leaves_z_offset after translation.
    leaves_z_offset
        Fixed additional z-lift for leaves-tagged bodies (after translations).
    """

    affects_spec = False
    needs_ctx = False

    def __init__(
        self,
        name_prefix: str = "vine",
        # Base (cluster center) shift bounds:
        pos_lo: Sequence[float] = (0.00, -1.00, 0.40),
        pos_hi: Sequence[float] = (1.00,  1.00, 0.80),
        # Per-body deviation (Normal(0, dev_std)):
        dev_std: Sequence[float] | float = (0.03, 0.03, 0.03),
        # Rotation controls:
        rot_enabled: bool = True,
        cluster_rot_angle_range: tuple[float, float] = (-0.05, 0.05),
        rot_angle_range: tuple[float, float] = (-0.15, 0.15),
        yaw_only: bool = False,
        # Leaves nicety:
        leaves_tag: str = "leaves",
        leaves_z_offset: float = 0.06,
    ):
        self.prefix = str(name_prefix)

        self.pos_lo = np.asarray(pos_lo, dtype=float)
        self.pos_hi = np.asarray(pos_hi, dtype=float)

        self.dev_std = np.asarray(dev_std, dtype=float)
        if self.dev_std.ndim == 0:
            self.dev_std = np.array([float(self.dev_std)] * 3, dtype=float)

        self.rot_enabled = rot_enabled
        self.cluster_rot_lo, self.cluster_rot_hi = cluster_rot_angle_range
        self.rot_lo, self.rot_hi = rot_angle_range
        self.yaw_only = yaw_only

        self.leaves_tag = leaves_tag
        self.leaves_z_offset = float(leaves_z_offset)

    # --------------------------------------------------------------------- #

    def apply(self, *, spec, model, data, rng, ctx=None):
        # ---- sample the episode-wise shared base translation (cluster center)
        base_shift = rng.uniform(self.pos_lo, self.pos_hi)

        # ---- sample a shared base rotation (optional)
        q_base = None
        if self.rot_enabled:
            axis_b, angle_b = _random_axis_angle(
                rng, self.cluster_rot_lo, self.cluster_rot_hi, self.yaw_only
            )
            q_base = _axis_angle_to_quat(axis_b, angle_b)

        # ---- apply to each matching body
        for bid in range(model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
            if not name.startswith(self.prefix):
                continue

            # ----- position: base shift + per-body deviation (Normal)
            deviation = rng.normal(loc=0.0, scale=self.dev_std, size=3)
            model.body_pos[bid] += base_shift + deviation

            # Leaves get an extra z-lift
            if self.leaves_tag and self.leaves_tag in name:
                model.body_pos[bid][2] += self.leaves_z_offset

            # ----- rotation: shared base rotation, then per-body rotation
            if self.rot_enabled:
                if q_base is not None:
                    model.body_quat[bid] = _quat_mul(q_base, model.body_quat[bid])

                axis, angle = _random_axis_angle(
                    rng, self.rot_lo, self.rot_hi, self.yaw_only
                )
                q_delta = _axis_angle_to_quat(axis, angle)
                model.body_quat[bid] = _quat_mul(q_delta, model.body_quat[bid])
