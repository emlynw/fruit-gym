import numpy as np
import os
from pathlib import Path
from typing import List, Optional, Union, Tuple
import yaml
import mujoco
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
from gymnasium.spaces import Box, Dict
from scipy.spatial.transform import Rotation

# External helpers (clean refactor)
from .constants import (
    PANDA_HOME,
    GRIPPER_HOME,
    GRIPPER_MIN,
    GRIPPER_MAX,
    PANDA_XYZ,
    CARTESIAN_BOUNDS,
    ROTATION_BOUNDS,
    default_obj_pos,
    gripper_sleep,
    grasp_threshold,
    ripe_mats,
)
from .control import (
    handle_gripper_control,
    run_opspace_for_duration,
    run_opspace_substeps,
)
from .observation import get_obs
from .indexing import index_fruits_and_stems
from .reward import compute_reward, RewardConfig
from .mujoco_utils import tick_removal_timers
from fruit_gym.controllers.opspace import opspace
from fruit_gym.randomisers.factory import build_randomisers


# ---------------------------
# Utilities
# ---------------------------

def load_config(config_path: Union[str, Path]):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------
# Environment
# ---------------------------

class TestEnv(MujocoEnv, utils.EzPickle):
    r"""Clean refactor of the TestEnv using shared helpers for control, obs, and indexing."""

    metadata = {"render_modes": ["human", "rgb_array", "depth_array"]}

    def __init__(
        self,
        image_obs: bool = True,
        randomize_domain: bool = True,
        ee_dof: int = 6,  # 3 for position, 3 for orientation
        control_dt: float = 0.05,
        physics_dt: float = 0.002,
        width: int = 480,
        height: int = 480,
        pos_scale: float = 0.008,
        rot_scale: float = 0.5,
        cameras: Optional[List[str]] = None,
        reward_type: str = "dense",
        gripper_pause: bool = False,
        discrete_gripper: bool = True,
        render_mode: str = "rgb_array",
        config_path: Optional[Union[str, Path]] = None,
        **kwargs,
    ):
        utils.EzPickle.__init__(self, image_obs=image_obs, **kwargs)

        # Basic paths and timing
        p = Path(__file__).parent
        self.xml_path = os.path.join(p, "xmls")
        self.scene_path = os.path.join(self.xml_path, "scene.xml")
        self._n_substeps = int(float(control_dt) / float(physics_dt))
        self.frame_skip = 1

        # Defaults
        if cameras is None:
            cameras = ["wrist1", "wrist2"]

        # Public config
        self.image_obs = image_obs
        self.randomize_domain = randomize_domain
        self.ee_dof = ee_dof
        self.render_mode = render_mode
        self.width = width
        self.height = height
        self.pos_scale = pos_scale
        self.rot_scale = rot_scale
        self.cameras = cameras
        self.reward_type = reward_type
        self.gripper_pause = gripper_pause
        self.discrete_gripper = discrete_gripper

        self.reward_cfg = RewardConfig(reward_type=self.reward_type)  # tweak later if you want
        self._pending_removals = {}
        self._grasped_pending = set()
        self._blocks_picked = 0
        self._prev_phi_red = 0.0
        self._prev_phi_align = 0.0

        # Load YAML cfg (domain randomization, etc.)
        if config_path is None:
            config_path = Path(__file__).parent.parent / "configs" / "multi_strawb.yaml"
        self.cfg = load_config(config_path)

        # Build randomisers once
        self._randomisers = build_randomisers(self.cfg, xml_dir=self.xml_path)

        # Observation space (state-only here; image keys added below if needed)
        state_space = {
            "tcp_pose": Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
            "tcp_vel": Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
            "gripper_pos": Box(-1, 1, shape=(1,), dtype=np.float32),
        }
        if self.discrete_gripper:
            state_space["gripper_vec"] = Box(0, 1, shape=(4,), dtype=np.float32)

        self.observation_space = Dict({"state": Dict(state_space)})
        if image_obs:
            self.observation_space["images"] = Dict()
            for cam in self.cameras:
                self.observation_space["images"][cam] = Box(
                    0, 255, shape=(self.height, self.width, 3), dtype=np.uint8
                )

        # Initialise MujocoEnv
        MujocoEnv.__init__(
            self,
            self.scene_path,
            self.frame_skip,
            observation_space=self.observation_space,
            render_mode=self.render_mode,
            width=self.width,
            height=self.height,
            **kwargs,
        )
        self.model.opt.timestep = physics_dt

        # Action space: xyz + rpy + gripper (depending on ee_dof)
        self.action_space = Box(
            np.array([-1.0] * (self.ee_dof + 1), dtype=np.float32),
            np.array([1.0] * (self.ee_dof + 1), dtype=np.float32),
            dtype=np.float32,
        )

        # Offscreen viewers per camera
        self._viewers = {
            cam: MujocoRenderer(
                self.model,
                self.data,
                width=self.width,
                height=self.height,
                camera_name=cam,
            )
            for cam in self.cameras
        }

        # Internal set-up
        self._bootstrap_constants()
        self._index_handles()
        self._init_state()

    # ---------------------------
    # Internal helpers
    # ---------------------------

    def _bootstrap_constants(self) -> None:
        """Mirror constants to instance attributes used by helpers for compatibility."""
        # Home & limits
        self._PANDA_HOME = PANDA_HOME.astype(np.float32)
        self._GRIPPER_HOME = GRIPPER_HOME.astype(np.float32)
        self._GRIPPER_MAX = float(GRIPPER_MAX)
        self._GRIPPER_MIN = float(GRIPPER_MIN)
        self._CARTESIAN_BOUNDS = CARTESIAN_BOUNDS.astype(np.float32)
        self._ROTATION_BOUNDS = ROTATION_BOUNDS.astype(np.float32)
        self.ripe_mats = set(ripe_mats)
        # Timings / thresholds
        self.gripper_sleep = float(gripper_sleep)
        self.grasp_threshold = float(grasp_threshold)

    def _index_handles(self) -> None:
        self._panda_dof_ids = np.array([self.model.joint(f"joint{i}").id for i in range(1, 8)])
        self._panda_ctrl_ids = np.array([self.model.actuator(f"actuator{i}").id for i in range(1, 8)])
        self._gripper_ctrl_id = self.model.actuator("fingers_actuator").id
        self._pinch_site_id = self.model.site("pinch").id

    def _init_state(self) -> None:
        # Gripper UI vector
        self.gripper_dict = {
            "open": np.array([1, 0, 0, 0], dtype=np.float32),
            "closed": np.array([0, 1, 0, 0], dtype=np.float32),
            "opening": np.array([0, 0, 1, 0], dtype=np.float32),
            "closing": np.array([0, 0, 0, 1], dtype=np.float32),
        }
        self.prev_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self.prev_grasp_time = 0.0
        self.prev_grasp = 0.0
        self.gripper_state = 0  # 0=open, 1=closed
        self.prev_gripper_state = 0
        self.gripper_blocked = False

        # Cache camera/light defaults for DR
        for cam in self.cameras:
            setattr(self, f"{cam}_pos", self.model.body_pos[self.model.body(cam).id].copy())
            setattr(self, f"{cam}_quat", self.model.body_quat[self.model.body(cam).id].copy())
        self.init_light_pos = self.model.body_pos[self.model.body("light0").id].copy()

        # Reference EE orientation (kept for rotation clipping)
        self.initial_position = np.array([0.1, 0.0, 0.75], dtype=np.float32)
        self.initial_orientation = np.array([0.725, 0.0, 0.688, 0.0], dtype=np.float32)
        self.initial_rotation = Rotation.from_quat(self.initial_orientation)

        # Viewer lighting defaults (for potential DR hooks)
        self.init_headlight_diffuse = self.model.vis.headlight.diffuse.copy()
        self.init_headlight_ambient = self.model.vis.headlight.ambient.copy()
        self.init_headlight_specular = self.model.vis.headlight.specular.copy()

        # Put robot at home
        self._reset_arm_and_gripper()

    # ---------------------------
    # MujocoEnv overrides
    # ---------------------------

    def _initialize_simulation(self) -> Tuple[mujoco.MjModel, mujoco.MjData]:
        """Parent calls this; build model from XML path (no spec editing here)."""
        self._base_spec = mujoco.MjSpec.from_file(self.scene_path)
        self._mj_spec = self._base_spec.copy()
        model = self._base_spec.compile()
        model.vis.global_.offwidth = self.width
        model.vis.global_.offheight = self.height
        data = mujoco.MjData(model)
        return model, data

    def reset_model(self):
        # Robust reset with randomisation passes and opspace settle
        attempt = 0
        while True:
            attempt += 1
            self._mj_spec = self._base_spec.copy()

            # Randomisers that affect spec first
            viewer = self.mujoco_renderer._get_viewer("rgb_array")
            ctx = viewer.con
            for r in self._randomisers:
                if r.affects_spec:
                    r.apply(spec=self._mj_spec, model=None, data=None, rng=self.np_random, ctx=ctx)

            # Recompile and fresh data
            self.model = self._mj_spec.compile()
            self.data = mujoco.MjData(self.model)
            self._index_handles()
            self._reset_arm_and_gripper()

            # Randomisers that affect compiled model/data
            for r in self._randomisers:
                if not r.affects_spec:
                    r.apply(spec=None, model=self.model, data=self.data, rng=self.np_random, ctx=ctx)

            # Rebuild viewers for new model
            self._viewers = {
                cam: MujocoRenderer(
                    self.model, self.data, width=self.width, height=self.height, camera_name=cam
                )
                for cam in self.cameras
            }

            # Clean state and forward once
            self.data.qvel[:] = 0
            self.data.qacc[:] = 0
            self.data.qfrc_applied[:] = 0
            self.data.xfrc_applied[:] = 0
            mujoco.mj_forward(self.model, self.data)

            # Optionally fix mocap to nominal
            if not self.randomize_domain:
                self.data.mocap_pos[0] = self.initial_position
                self.data.mocap_quat[0] = np.roll(self.initial_orientation, 1)

            desired_pos = self.data.mocap_pos[0].copy()
            desired_quat = self.data.mocap_quat[0].copy()

            # Let opspace settle for a bit
            for _ in range(10 * self._n_substeps):
                tau = opspace(
                    model=self.model,
                    data=self.data,
                    site_id=self._pinch_site_id,
                    dof_ids=self._panda_dof_ids,
                    pos=self.data.mocap_pos[0],
                    ori=self.data.mocap_quat[0],
                    joint=self._PANDA_HOME,
                    gravity_comp=True,
                )
                self.data.ctrl[self._panda_ctrl_ids] = tau
                mujoco.mj_step(self.model, self.data)

            # Reset gripper markers
            self.grasp = -1.0
            self.prev_grasp_time = 0.0
            self.prev_gripper_state = 0
            self.gripper_state = 0
            self.gripper_blocked = False

            # Validate pose
            current_pos = self.data.sensor("pinch_pos").data.copy()
            current_quat = self.data.sensor("pinch_quat").data.copy()
            if (
                np.any(np.isnan(current_pos))
                or np.any(np.isnan(current_quat))
                or np.any(np.isinf(current_pos))
                or np.any(np.isinf(current_quat))
            ):
                continue

            pos_diff = np.linalg.norm(current_pos - desired_pos)
            cq = current_quat / np.linalg.norm(current_quat)
            dq = desired_quat / np.linalg.norm(desired_quat)
            dot = float(np.clip(abs(np.dot(cq, dq)), -1.0, 1.0))
            orient_diff = 2 * np.arccos(dot)

            if pos_diff < 0.1 and orient_diff < 0.2:
                # Index fruits & stems via shared helper
                (
                    fruit_instances,
                    ripe_ids,
                    unripe_ids,
                    stem_to_fruit,
                ) = index_fruits_and_stems(self.model, ripe_mats=self.ripe_mats)
                 # Normalise instance keys to ints (names like stem_1_2 don't matter; we use geom ids)
                self.fruit_instances = {int(fid): inst for fid, inst in fruit_instances.items()}
                self.ripe_ids = [int(i) for i in ripe_ids]
                self.unripe_ids = [int(i) for i in unripe_ids]
                self.stem_to_fruit = {int(k): int(v) for k, v in stem_to_fruit.items()}

                # Map ripe/unripe to red/green lists used by reward/deletion
                self.red_blocks   = sorted(self.ripe_ids)
                self.green_blocks = sorted(self.unripe_ids)

                # Cache initial positions via representative fruit geom (first fruit geom)
                self.red_positions   = {}
                self.green_positions = {}

                for fid in self.red_blocks:
                    inst = self.fruit_instances.get(fid)
                    if inst and getattr(inst, "fruit_geoms", None):
                        gid = int(inst.fruit_geoms[0])
                        self.red_positions[fid] = self.data.geom_xpos[gid].copy()

                for fid in self.green_blocks:
                    inst = self.fruit_instances.get(fid)
                    if inst and getattr(inst, "fruit_geoms", None):
                        gid = int(inst.fruit_geoms[0])
                        self.green_positions[fid] = self.data.geom_xpos[gid].copy()

                # Reset disappearance & potential-shaping bookkeeping
                self._pending_removals = {}
                self._grasped_pending = set()
                self._blocks_picked = 0
                self._prev_phi_red = 0.0
                self._prev_phi_align = 0.0

                return get_obs(self)
            else:
                # print(f"Reset attempt {attempt}: pos={pos_diff:.4f} ori={orient_diff:.4f} -> retry")
                if attempt > 100:
                    raise RuntimeError("Failed to achieve valid reset after multiple attempts")

    def step(self, action):
        # Validate & clip
        if np.array(action).shape != self.action_space.shape:
            raise ValueError("Action dimension mismatch")
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Unpack action (end-effector coordinates are expressed in the EE frame)
        if self.ee_dof == 3:
            z, y, x, grasp = action
            drot = None
        elif self.ee_dof == 4:
            z, y, x, yaw, grasp = action
            drot = np.array([0.0, 0.0, yaw]) * self.rot_scale
        elif self.ee_dof == 6:
            z, y, x, roll, pitch, yaw, grasp = action
            drot = np.array([roll, pitch, yaw]) * self.rot_scale
        else:
            raise ValueError("ee_dof must be 3, 4, or 6")

        # Position update in world via current EE orientation
        dpos_local = np.array([x, y, z], dtype=np.float32) * self.pos_scale
        current_quat_xyzw = np.roll(self.data.sensor("pinch_quat").data, -1)
        current_rot = Rotation.from_quat(current_quat_xyzw)
        dpos_world = current_rot.apply(dpos_local)
        npos = np.clip(self.data.sensor("pinch_pos").data + dpos_world, *self._CARTESIAN_BOUNDS)
        self.data.mocap_pos[0] = npos

        # Orientation update (relative to initial orientation, with bounds)
        if drot is not None:
            current_rot = Rotation.from_quat(current_quat_xyzw)
            action_rot = Rotation.from_euler("xyz", drot)
            new_rot = action_rot * current_rot
            rel_rot = self.initial_rotation.inv() * new_rot
            rel_euler = rel_rot.as_euler("xyz")
            clipped = np.clip(rel_euler, self._ROTATION_BOUNDS[0], self._ROTATION_BOUNDS[1])
            final_rot = self.initial_rotation * Rotation.from_euler("xyz", clipped)
            self.data.mocap_quat[0] = np.roll(final_rot.as_quat(), 1)  # xyzw -> wxyz

        # Gripper control via helper
        moving_gripper, target_t = handle_gripper_control(self, action)

        # Physics integration: either hold until gripper motion done, or do substeps with opspace
        if self.discrete_gripper and self.gripper_pause and moving_gripper:
            run_opspace_for_duration(self, until_time=target_t)
        else:
            run_opspace_substeps(self, n_substeps=self._n_substeps, warmup_ratio=0.2)

        # Observation via shared helper
        obs = get_obs(self)
        if self.render_mode == "human":
            self.render()

        # Reward
        reward, info = compute_reward(self, action)
        # make due fruit disappear
        tick_removal_timers(self)

        if self.reward_cfg.reward_type == "sparse":
            info["dense_reward"] = reward
            reward = 1.0 if info.get("r_grasp", 0.0) > 0 else 0.0
        terminated = bool(info.get("success", False))
        self.prev_gripper_state = self.gripper_state
        self.prev_action = action.copy()
        return obs, reward, terminated, False, info

    def render(self):
        return [self._viewers[c].render("rgb_array") for c in self.cameras]

    # ---------------------------
    # Robot reset helpers
    # ---------------------------

    def _reset_arm_and_gripper(self) -> None:
        self.data.qpos[self._panda_dof_ids] = self._PANDA_HOME
        self.data.qpos[7:9] = self._GRIPPER_HOME
        self.data.ctrl[self._gripper_ctrl_id] = self._GRIPPER_MAX
        self.gripper_vec = self.gripper_dict["open"]
        mujoco.mj_forward(self.model, self.data)
        self.data.mocap_pos[0] = self.data.sensor("pinch_pos").data.copy()
        self.data.mocap_quat[0] = self.data.sensor("pinch_quat").data.copy()
        mujoco.mj_step(self.model, self.data)
