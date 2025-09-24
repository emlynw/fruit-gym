# fruit_gym/envs/observation.py

from __future__ import annotations
from typing import Dict as TDict, Any
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from .reward import get_privileged_info


def get_vel(env) -> np.ndarray:
    """
    Cartesian (6,) velocity of the EE at the pinch site: [vx,vy,vz, wx,wy,wz].
    """
    dq = env.data.qvel[env._panda_dof_ids]
    J_v = np.zeros((3, env.model.nv), dtype=np.float64)
    J_w = np.zeros((3, env.model.nv), dtype=np.float64)
    mujoco.mj_jacSite(env.model, env.data, J_v, J_w, env._pinch_site_id)
    J_v, J_w = J_v[:, env._panda_dof_ids], J_w[:, env._panda_dof_ids]
    J = np.vstack((J_v, J_w))
    dx = J @ dq
    return dx.astype(np.float32)


def get_obs(env) -> TDict[str, Any]:
    """
    Build the observation dict:
      - obs["state"]["tcp_pose"]: [pos(3), quat_xyzw(4)]
      - obs["state"]["tcp_vel"]
      - obs["state"]["gripper_pos"]
      - obs["state"]["gripper_vec"] (if discrete)
      - obs["images"][cam] if image_obs
      - optionally merges extra state from env._get_strawberry_state_obs(tcp_pos) if available and image_obs=False
    """
    obs: TDict[str, Any] = {"state": {}}

    # --- TCP pose ---
    tcp_world_pos = env.data.sensor("pinch_pos").data.copy()
    tcp_world_quat_xyzw = np.roll(env.data.sensor("pinch_quat").data, -1).copy()  # mujoco wxyz -> xyzw

    if getattr(env, "randomize_domain", False):
        # position noise
        position_noise_std = env.cfg.get("domain_randomization", {}).get("ee_pos_noise_std", 0.005)
        tcp_world_pos += env.np_random.normal(0, position_noise_std, size=3)

        # orientation noise (small axis-angle)
        orientation_noise_std = env.cfg.get("domain_randomization", {}).get("ee_ori_noise_std", 0.005)
        axis_angle = env.np_random.normal(0, orientation_noise_std, size=3)
        small_rot = Rotation.from_rotvec(axis_angle)
        current_rot = Rotation.from_quat(tcp_world_quat_xyzw)
        tcp_world_quat_xyzw = (small_rot * current_rot).as_quat()

    obs["state"]["tcp_pose"] = np.concatenate([tcp_world_pos, tcp_world_quat_xyzw]).astype(np.float32)
    obs["state"]["tcp_vel"] = get_vel(env)
    # Normalize to ~[0, 1] using your prior convention (2 * qpos[8] / GRIPPER_HOME)
    obs["state"]["gripper_pos"] = np.array([2 * env.data.qpos[8] / env._GRIPPER_HOME[0]], dtype=np.float32)
    if env.discrete_gripper:
        obs["state"]["gripper_vec"] = env.gripper_vec.astype(np.float32)

    # --- Privileged (critic-only) ---
    if getattr(env, "include_privileged_obs", False):
        # Cache-once-per-step if available; otherwise compute here
        priv = getattr(env, "current_privileged_info", None)
        if priv is None:
            priv = get_privileged_info(env)
        obs["priv_state"] = {
            "min_red_distance":              np.array([priv["min_red_dist"]], dtype=np.float32),
            "gripper_alignment_quality":     np.array([priv["radial_dist"]], dtype=np.float32),
            "good_grasp_detected":           np.array([float(priv["good_grasp"])], dtype=np.float32),
            "bad_grasp_detected":            np.array([float(priv["bad_grasp"])], dtype=np.float32),
            "collision_detected":            np.array([float(priv["collision_detected"])], dtype=np.float32),
            "red_stems_in_box_count":        np.array([priv["red_stems_in_box_count"]], dtype=np.float32),
            "green_stems_in_box_count":      np.array([priv["green_stems_in_box_count"]], dtype=np.float32),
            "left_finger_contacts":          np.array([priv["left_finger_contacts"]], dtype=np.float32),
            "right_finger_contacts":         np.array([priv["right_finger_contacts"]], dtype=np.float32),
            "total_distractor_displacement": np.array([priv["total_displacement"]], dtype=np.float32),
        }

    # --- Images ---
    if env.image_obs:
        obs["images"] = {}
        for cam_name in env.cameras:
            obs["images"][cam_name] = env._viewers[cam_name].render(render_mode="rgb_array")

    # --- Optional extra state (non-image mode) ---
    if not env.image_obs and hasattr(env, "_get_strawberry_state_obs"):
        try:
            extra = env._get_strawberry_state_obs(tcp_world_pos)
            if isinstance(extra, dict):
                obs["state"].update(extra)
        except Exception:
            # Keep observation robust even if the helper throws
            pass

    # Human viewer support (match your prior behavior)
    if env.render_mode == "human" and "wrist1" in getattr(env, "_viewers", {}):
        env._viewers["wrist1"].render(env.render_mode)

    return obs
