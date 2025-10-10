# fruit_gym/envs/control.py

from __future__ import annotations
from typing import Tuple
import numpy as np
import mujoco

from fruit_gym.controllers.opspace import opspace


def handle_gripper_control(env, action: np.ndarray) -> Tuple[bool, float]:
    """
    Discrete/continuous gripper control, matching your previous semantics.

    Returns:
        moving_gripper (bool): True if a discrete open/close command was just issued.
        target_sim_time (float): If moving_gripper is True and pause is desired,
                                 the simulation time to run-until; else 0.0.
    """
    moving_gripper = False
    target_sim_time = 0.0

    if env.discrete_gripper:
        grasp = action[-1]
        if env.data.time - env.prev_grasp_time < env.gripper_sleep:
            # Ignore new commands during sleep window
            env.gripper_blocked = True
            grasp = env.prev_grasp
        else:
            if grasp <= env.grasp_threshold and env.gripper_state == 0:
                # already open → keep open marker
                env.gripper_vec = env.gripper_dict["open"]
                env.gripper_blocked = False

            elif grasp >= -env.grasp_threshold and env.gripper_state == 1:
                # already closed → keep closed marker
                env.gripper_vec = env.gripper_dict["closed"]
                env.gripper_blocked = False

            elif grasp < -env.grasp_threshold and env.gripper_state == 1:
                # command: OPEN
                env.data.ctrl[env._gripper_ctrl_id] = env._GRIPPER_MAX
                env.gripper_state = 0
                env.gripper_vec = env.gripper_dict["opening"]
                env.prev_grasp_time = env.data.time
                env.prev_grasp = grasp
                env.gripper_blocked = True
                moving_gripper = True
                target_sim_time = env.data.time + env.gripper_sleep

            elif grasp > env.grasp_threshold and env.gripper_state == 0:
                # command: CLOSE
                env.data.ctrl[env._gripper_ctrl_id] = 0.0
                env.gripper_state = 1
                env.gripper_vec = env.gripper_dict["closing"]
                env.prev_grasp_time = env.data.time
                env.prev_grasp = grasp
                env.gripper_blocked = True
                moving_gripper = True
                target_sim_time = env.data.time + env.gripper_sleep

    else:
        # Continuous gripper control
        grasp_action = action[-1]
        gripper_speed = 0.005
        current_pos = env.data.qpos[env._gripper_ctrl_id]
        new_target_pos = current_pos + -grasp_action * gripper_speed
        env.data.ctrl[env._gripper_ctrl_id] = np.clip(new_target_pos, 0.0, env._GRIPPER_MAX)

    return moving_gripper, target_sim_time


def run_opspace_for_duration(env, until_time: float) -> None:
    """
    Hold the end-effector at env.data.mocap_{pos,quat}[0] using opspace until env.data.time >= until_time.
    """
    while env.data.time < until_time:
        tau = opspace(
            model=env.model,
            data=env.data,
            site_id=env._pinch_site_id,
            dof_ids=env._panda_dof_ids,
            pos=env.data.mocap_pos[0],
            ori=env.data.mocap_quat[0],
            joint=env._PANDA_HOME,
            gravity_comp=True,
            prev_tau_des=env.prev_tau_des,
        )
        env.data.ctrl[env._panda_ctrl_ids] = tau
        env.prev_tau_des = tau.copy()
        mujoco.mj_step(env.model, env.data)


def run_opspace_substeps(env, n_substeps: int, warmup_ratio: float = 0.0) -> None:
    """
    early substeps do nothing, later ones apply opspace.
    """
    warmup = int(max(0, min(n_substeps, int(n_substeps * warmup_ratio))))
    for i in range(n_substeps):
        if i < warmup:
            # let Mujoco integrate without torque to avoid jerk
            mujoco.mj_step(env.model, env.data)
        else:
            tau = opspace(
                model=env.model,
                data=env.data,
                site_id=env._pinch_site_id,
                dof_ids=env._panda_dof_ids,
                pos=env.data.mocap_pos[0],
                ori=env.data.mocap_quat[0],
                joint=env._PANDA_HOME,
                gravity_comp=True,
                prev_tau_des=env.prev_tau_des,
            )
            env.data.ctrl[env._panda_ctrl_ids] = tau
            env.prev_tau_des = tau.copy()
            mujoco.mj_step(env.model, env.data)
