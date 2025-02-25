from typing import Optional, Tuple, Union
import mujoco
import numpy as np
from dm_robotics.transformations import transformations as tr

def pseudo_inverse(M, damped=False, lambda_=0.2):
    if not damped:
        return np.linalg.pinv(M)
    U, sing_vals, V = np.linalg.svd(M, full_matrices=False)
    S_inv = sing_vals / (sing_vals ** 2 + lambda_ ** 2)
    return (V.T * S_inv) @ U.T

def pd_control(
    x: np.ndarray,
    x_des: np.ndarray,
    dx: np.ndarray,
    kp_kv: np.ndarray,
    max_pos_error: float = 0.05,
) -> np.ndarray:
    x_err = np.clip(x - x_des, -max_pos_error, max_pos_error)
    dx_err = dx
    return -kp_kv[:, 0] * x_err - kp_kv[:, 1] * dx_err

def pd_control_orientation(
    quat: np.ndarray,
    quat_des: np.ndarray,
    w: np.ndarray,
    kp_kv: np.ndarray,
    max_ori_error: float = 0.05,
) -> np.ndarray:
    quat_err = tr.quat_diff_active(source_quat=quat_des, target_quat=quat)
    ori_err = np.clip(tr.quat_to_axisangle(quat_err), -max_ori_error, max_ori_error)
    w_err = w
    return -kp_kv[:, 0] * ori_err - kp_kv[:, 1] * w_err

def saturate_torque_rate(tau_d_calculated, tau_J_d, delta_tau_max):
    return tau_J_d + np.clip(tau_d_calculated - tau_J_d, -delta_tau_max, delta_tau_max)

def opspace(
    model,
    data,
    site_id,
    dof_ids: np.ndarray,
    pos: Optional[np.ndarray] = None,
    ori: Optional[np.ndarray] = None,
    joint: Optional[np.ndarray] = None,
    pos_gains: Union[Tuple[float, float, float], np.ndarray] = (2000.0, 2000.0, 2000.0),
    ori_gains: Union[Tuple[float, float, float], np.ndarray] = (150.0, 150.0, 150.0),
    joint_upper_limits: Union[Tuple[float, float, float, float, float, float, float], np.ndarray] = (2.75, 1.58, 2.75, -0.15, 2.75, 3.66, 2.75),
    joint_lower_limits: Union[Tuple[float, float, float, float, float, float, float], np.ndarray] = (-2.75, -1.58, -2.75, -2.99, -2.75, -0.11, -2.75),
    translational_damping: float = 89.0,
    rotational_damping: float = 6.0,
    nullspace_stiffness: float = 0.5,
    nullspace_damping: float = 1.414,
    joint1_nullspace_stiffness: float = 10.0,
    max_pos_error: float = 0.01,
    max_ori_error: float = 0.03,
    delta_tau_max: float = 1.0,
    gravity_comp: bool = True,
    damped: bool = False,
    lambda_: float = 0.2,
) -> np.ndarray:
    
    
    x_des = data.site_xpos[site_id] if pos is None else np.asarray(pos)
    quat_des = (
        tr.mat_to_quat(data.site_xmat[site_id].reshape((3, 3))) if ori is None else
        tr.mat_to_quat(np.asarray(ori)) if ori.shape == (3, 3) else
        ori
    )
    q_d_nullspace = data.qpos[dof_ids] if joint is None else np.asarray(joint)

    kp_pos = np.asarray(pos_gains)
    kd_pos = translational_damping * np.ones_like(kp_pos)
    kp_kv_pos = np.stack([kp_pos, kd_pos], axis=-1)

    kp_ori = np.asarray(ori_gains)
    kd_ori = rotational_damping * np.ones_like(kp_ori)
    kp_kv_ori = np.stack([kp_ori, kd_ori], axis=-1)

    q = data.qpos[dof_ids]
    dq = data.qvel[dof_ids]

    J_v = np.zeros((3, model.nv), dtype=np.float64)
    J_w = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jacSite(model, data, J_v, J_w, site_id)
    J_v, J_w = J_v[:, dof_ids], J_w[:, dof_ids]
    J = np.vstack((J_v, J_w))

    x = data.site_xpos[site_id]
    dx = J_v @ dq
    ddx = pd_control(x, x_des, dx, kp_kv_pos, max_pos_error)

    quat = tr.mat_to_quat(data.site_xmat[site_id].reshape((3, 3)))
    if quat @ quat_des < 0.0:
        quat *= -1.0
    w = J_w @ dq
    dw = pd_control_orientation(quat, quat_des, w, kp_kv_ori, max_ori_error)

    C = data.qfrc_bias[dof_ids]
    F = np.concatenate([ddx, dw])
    tau_task = J.T @ F

    q_error = q_d_nullspace - q
    q_error[0] *= joint1_nullspace_stiffness
    dq_error = dq
    dq_error[0] *= 2 * np.sqrt(joint1_nullspace_stiffness)
    tau_nullspace = nullspace_stiffness * q_error - nullspace_damping * dq_error

    if damped:
        jacobian_transpose_pinv = pseudo_inverse(J.T, damped=True, lambda_=lambda_)
        tau_nullspace = (np.eye(len(dof_ids)) - J.T @ jacobian_transpose_pinv) @ tau_nullspace

    tau = tau_task + tau_nullspace

    # Check if any joint is at or near its limits and adjust torques
    for i, (q_i, q_upper, q_lower) in enumerate(zip(q, joint_upper_limits, joint_lower_limits)):
        if q_i >= q_upper and tau[i] > 0.0:
            tau[i] = 0.0
        elif q_i <= q_lower and tau[i] < 0.0:
            tau[i] = 0.0

    if gravity_comp:
        tau += C

    return saturate_torque_rate(tau, data.qfrc_actuator[dof_ids], delta_tau_max)
