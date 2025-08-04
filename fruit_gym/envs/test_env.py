import numpy as np
import os
from pathlib import Path
from typing import List, Optional, Union
import yaml
import mujoco
import random
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
from gymnasium.spaces import Box, Dict
from scipy.spatial.transform import Rotation
from fruit_gym.controllers.opspace import opspace
from fruit_gym.randomisers.factory import build_randomisers
import copy

def load_config(config_path):
    with open(config_path, 'r') as config_file:
        return yaml.safe_load(config_file)

class TestEnv(MujocoEnv, utils.EzPickle):
    r""" Testing refactor
    """


    metadata = { 
        "render_modes": ["human", "rgb_array", "depth_array"], 
    }
    
    def __init__(
        self,
        image_obs: bool = True,
        randomize_domain: bool = True,
        ee_dof: int = 6, # 3 for position, 3 for orientation
        control_dt: float = 0.05,
        physics_dt: float = 0.002,
        width: int = 480,
        height: int = 480,
        pos_scale: float = 0.008,
        rot_scale: float = 0.5,
        cameras: List[str] = None,
        reward_type: str = "dense",
        gripper_pause: bool = False,
        discrete_gripper: bool = True,
        render_mode: str = "rgb_array",
        config_path: Optional[Union[str, Path]] = None,
        **kwargs,
    ):
        utils.EzPickle.__init__(self, image_obs=image_obs, **kwargs)
        p = Path(__file__).parent
        self.xml_path = os.path.join(p, "xmls")
        scene_path = os.path.join(self.xml_path, "scene.xml")
        self.scene_path = scene_path
        self._n_substeps = int(float(control_dt) / float(physics_dt))
        self.frame_skip = 1

        if cameras is None:
            cameras = ["wrist1", "wrist2"]

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

        self._PANDA_HOME = np.array([0.0, -1.6, 0.0, -2.54, -0.05, 2.49, 0.822], dtype=np.float32)
        self._GRIPPER_HOME = np.array([0.0141, 0.0141], dtype=np.float32)
        self._GRIPPER_MIN = 0
        self._GRIPPER_MAX = 0.004
        self._PANDA_XYZ = np.array([0.1, 0, 0.8], dtype=np.float32)
        self._CARTESIAN_BOUNDS = np.array([[0.05, -0.2, 0.6], [0.55, 0.2, 0.95]], dtype=np.float32)
        self._ROTATION_BOUNDS = np.array([[-np.pi/3, -np.pi/6, -np.pi/10],[np.pi/3, np.pi/6, np.pi/10]], dtype=np.float32)
        self.default_obj_pos = np.array([0.42, 0, 0.85])
        self.gripper_sleep = 0.6
        self.grasp_threshold = 0.333

        if config_path is None:
            config_path = Path(__file__).parent.parent / "configs" / "multi_strawb.yaml"
        self.cfg = load_config(config_path)

        # Load the domain randomization configuration
        self._randomisers = build_randomisers(self.cfg, xml_dir=self.xml_path)

        # Define the basic state space components
        state_space_dict = {
            "tcp_pose": Box(-np.inf, np.inf, shape=(7,), dtype=np.float32), # 3 pos, 4 quat (xyzw)
            "tcp_vel": Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
            "gripper_pos": Box(-1, 1, shape=(1,), dtype=np.float32),
        }
        if self.discrete_gripper:
            state_space_dict["gripper_vec"] = Box(0, 1, shape=(4,), dtype=np.float32)
        

        self.observation_space = Dict({"state": Dict(state_space_dict)})

        if image_obs:
            self.observation_space["images"] = Dict()
            for camera in self.cameras:
                self.observation_space["images"][camera] = Box(
                    0, 255, shape=(self.height, self.width, 3), dtype=np.uint8
                )

        MujocoEnv.__init__(
            self, 
            scene_path, 
            self.frame_skip, 
            observation_space=self.observation_space, 
            render_mode=self.render_mode,
            width=self.width,
            height=self.height, 
            **kwargs,
        )
        self.model.opt.timestep = physics_dt

        self.action_space = Box(
            np.array([-1.0]*(self.ee_dof+1)), 
            np.array([1.0]*(self.ee_dof+1)),
            dtype=np.float32,
        )
        self._viewers = {
            cam: MujocoRenderer(
                self.model,
                self.data,
                width=self.width,
                height=self.height,
                camera_name=cam,           # <‑‑ choose the camera here
            )
            for cam in self.cameras
        }
        self.setup()

    def _initialize_simulation(self):
        """Parent calls this; build model from XML path (no spec editing here)."""
        self._base_spec = mujoco.MjSpec.from_file(self.scene_path)
        self._mj_spec = self._base_spec.copy()
        model = self._base_spec.compile()
        model.vis.global_.offwidth = self.width
        model.vis.global_.offheight = self.height
        data = mujoco.MjData(model)
        return model, data

    def setup(self):

        self._panda_dof_ids = np.array([self.model.joint(f"joint{i}").id for i in range(1, 8)])
        self._panda_ctrl_ids = np.array([self.model.actuator(f"actuator{i}").id for i in range(1, 8)])
        self._gripper_ctrl_id = self.model.actuator("fingers_actuator").id
        self._pinch_site_id = self.model.site("pinch").id

        self.prev_action = np.zeros(self.action_space.shape)
        self.prev_grasp_time = 0.0
        self.prev_grasp = 0.0
        self.gripper_dict = {
            "open": np.array([1, 0, 0, 0], dtype=np.float32),
            "closed": np.array([0, 1, 0, 0], dtype=np.float32),
            "opening": np.array([0, 0, 1, 0], dtype=np.float32),
            "closing": np.array([0, 0, 0, 1], dtype=np.float32),
        }

        self.reset_arm_and_gripper()

        # Store initial values for randomization
        for camera_name in self.cameras:
            setattr(self, f"{camera_name}_pos", self.model.body_pos[self.model.body(camera_name).id].copy())
            setattr(self, f"{camera_name}_quat", self.model.body_quat[self.model.body(camera_name).id].copy())
        self.init_light_pos = self.model.body_pos[self.model.body('light0').id].copy()

        self.skybox_tex_ids = []
        self.floor_tex_ids = []

        self.initial_position = np.array([0.1, 0.0, 0.75], dtype=np.float32)
        self.initial_orientation = [0.725, 0.0, 0.688, 0.0]
        self.initial_rotation = Rotation.from_quat(self.initial_orientation)

        self.init_headlight_diffuse = self.model.vis.headlight.diffuse.copy()
        self.init_headlight_ambient = self.model.vis.headlight.ambient.copy()
        self.init_headlight_specular = self.model.vis.headlight.specular.copy()


    def _set_inactive_properties_recursive(self, body_id: int):
        """
        Recursively sets geoms under body_id to group 3 
        and makes them non-collidable
        """
        # Process geoms of the current body
        geom_start = self.model.body_geomadr[body_id]
        geom_count = self.model.body_geomnum[body_id]
        for k in range(geom_count):
            geom_id = geom_start + k
            self.model.geom_group[geom_id] = 3  # Assign to group 3
            self.model.geom_contype[geom_id] = 0
            self.model.geom_conaffinity[geom_id] = 0
        
        # Recurse for children
        for child_body_id in range(self.model.nbody):
            if self.model.body_parentid[child_body_id] == body_id:
                self._set_inactive_properties_recursive(child_body_id)

    def reset_arm_and_gripper(self):
        self.data.qpos[self._panda_dof_ids] = self._PANDA_HOME
        self.data.qpos[7:9] = self._GRIPPER_HOME
        self.data.ctrl[self._gripper_ctrl_id] = self._GRIPPER_MAX
        self.gripper_vec = self.gripper_dict["open"]
        mujoco.mj_forward(self.model, self.data)
        self.data.mocap_pos[0] = self.data.sensor("pinch_pos").data.copy()
        self.data.mocap_quat[0] = self.data.sensor("pinch_quat").data.copy()
        mujoco.mj_step(self.model, self.data)


    def reset_model(self):
        # Some random resets were getting mujoco Nan warnings that's why the loop
        attempt = 0
        while True:
            attempt += 1
            self._mj_spec = self._base_spec.copy()

            viewer = self.mujoco_renderer._get_viewer("rgb_array")
            ctx = viewer.con

             # -------- first pass: spec randomisers --------
            for r in self._randomisers:
                if r.affects_spec:
                    r.apply(spec=self._mj_spec, model=None, data=None, rng=self.np_random, ctx=ctx)

            # re-compile
            self.model = self._mj_spec.compile()
            self.data = mujoco.MjData(self.model)
            self.reset_arm_and_gripper()


            # -------- second pass: model/data randomisers --------
            for r in self._randomisers:
                if not r.affects_spec:
                    r.apply(spec=None, model=self.model, data=self.data, rng=self.np_random, ctx=ctx)

            self._viewers = {
            cam: MujocoRenderer(
                self.model,
                self.data,
                width=self.width,
                height=self.height,
                camera_name=cam,           # <‑‑ choose the camera here
            )
            for cam in self.cameras
            }

            self.data.qvel[:] = 0
            self.data.qacc[:] = 0
            self.data.qfrc_applied[:] = 0
            self.data.xfrc_applied[:] = 0
            mujoco.mj_forward(self.model, self.data)

            if not self.randomize_domain:
                self.data.mocap_pos[0] = self.initial_position
                self.data.mocap_quat[0] = np.roll(self.initial_orientation, 1)

            desired_pos = self.data.mocap_pos[0].copy()
            desired_quat = self.data.mocap_quat[0].copy()

            for _ in range(10*self._n_substeps):
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
            

            self.grasp = -1.0
            self.prev_grasp_time = 0.0
            self.prev_gripper_state = 0 # 0 for open, 1 for closed
            self.gripper_state = 0
            self.gripper_blocked = False

             # Get the current end-effector pose from sensors.
            current_pos = self.data.sensor("pinch_pos").data.copy()
            current_quat = self.data.sensor("pinch_quat").data.copy()

            # Check that sensor readings are finite.
            if (np.any(np.isnan(current_pos)) or np.any(np.isnan(current_quat)) or
                np.any(np.isinf(current_pos)) or np.any(np.isinf(current_quat))):
                continue

            # Compute the difference in position.
            pos_diff = np.linalg.norm(current_pos - desired_pos)
            # Compute orientation difference using the dot-product of unit quaternions.
            current_quat_norm = current_quat / np.linalg.norm(current_quat)
            desired_quat_norm = desired_quat / np.linalg.norm(desired_quat)
            dot = np.abs(np.dot(current_quat_norm, desired_quat_norm))
            dot = np.clip(dot, -1.0, 1.0)
            orient_diff = 2 * np.arccos(dot)

            pos_threshold = 0.1    
            orient_threshold = 0.2    

            if pos_diff < pos_threshold and orient_diff < orient_threshold:
                return self._get_obs()
            else:
                print(
                    f"Reset attempt {attempt+1}: pose error too high "
                    f"(pos_diff: {pos_diff:.4f}, orient_diff: {orient_diff:.4f}), retrying reset."
                )
                if attempt > 100:
                    raise RuntimeError("Failed to achieve valid reset after multiple attempts")


    def step(self, action):
        if np.array(action).shape != self.action_space.shape:
            raise ValueError("Action dimension mismatch")
        action = np.clip(action, self.action_space.low, self.action_space.high)
        # Scale actions (zyx because end effector frame z is along the gripper axis)
        if self.ee_dof == 3:
            z, y, x, grasp = action
        elif self.ee_dof == 4:
            z, y, x, yaw, grasp = action
            roll, pitch = 0, 0
            drot = np.array([roll, pitch, yaw]) * self.rot_scale
        elif self.ee_dof == 6:
            z, y, x, roll, pitch, yaw, grasp = action
            drot = np.array([roll, pitch, yaw]) * self.rot_scale
        dpos = np.array([x, y, z]) * self.pos_scale
        # Apply position change
        pos = self.data.sensor("pinch_pos").data
        current_quat = np.roll(self.data.sensor("pinch_quat").data, -1)
        current_rotation = Rotation.from_quat(current_quat)

        dpos_world = current_rotation.apply(dpos)
        npos = np.clip(pos + dpos_world, *self._CARTESIAN_BOUNDS)
        self.data.mocap_pos[0] = npos

        if self.ee_dof > 3:
            # Convert mujoco wxyz to scipy xyzw
            current_quat = np.roll(self.data.sensor("pinch_quat").data, -1)
            current_rotation = Rotation.from_quat(current_quat)
            # Convert the action rotation to a Rotation object
            action_rotation = Rotation.from_euler('xyz', drot)
            # Apply the action rotation
            new_rotation = action_rotation * current_rotation
            # Calculate the new relative rotation
            new_relative_rotation = self.initial_rotation.inv() * new_rotation
            # Convert to euler angles and clip
            relative_euler = new_relative_rotation.as_euler('xyz')
            clipped_euler = np.clip(relative_euler, self._ROTATION_BOUNDS[0], self._ROTATION_BOUNDS[1])
            # Convert back to rotation and apply to initial orientation
            clipped_rotation = Rotation.from_euler('xyz', clipped_euler)
            final_rotation = self.initial_rotation * clipped_rotation
            # Set the final orientation
            self.data.mocap_quat[0] = np.roll(final_rotation.as_quat(), 1)

        # --- Handle gripper and simulation ---
        moving_gripper, target_sim_time = self._handle_gripper_control(action)

        if self.discrete_gripper and self.gripper_pause and moving_gripper:
            while self.data.time < target_sim_time:
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
        else:
            for i in range(self._n_substeps):
                if i < self._n_substeps/5:
                    continue
                else:
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

        # Observation
        obs = self._get_obs()
        if self.render_mode == "human":
            self.render()

        # Reward
        reward, info = self._compute_reward(action)
        if info['success'] == True:
            terminated = True
        else:
            terminated = False
        self.prev_gripper_state = self.gripper_state

        return obs, reward, terminated, False, info 
    
    def render(self):
        rendered_frames = []
        for camera in self.cameras:
            rendered_frames.append(
                self._viewers[camera].render("rgb_array")
            )
        return rendered_frames
    
    def _handle_gripper_control(self, action: np.ndarray):
        """
        Determines gripper control value and state based on the action.
        This method supports both discrete and continuous gripper control.

        Args:
            action (np.ndarray): The action array from the policy.

        Returns:
            Tuple[bool, float]: A tuple containing:
                - moving_gripper (bool): True if a discrete grasp was initiated.
                - target_sim_time (float): The simulation time to pause for a discrete grasp.
        """
        moving_gripper = False
        target_sim_time = 0.0

        if self.discrete_gripper:
            grasp = action[-1]
            if self.data.time - self.prev_grasp_time < self.gripper_sleep:
                self.gripper_blocked = True
                grasp = self.prev_grasp
            else:
                if grasp <= self.grasp_threshold and self.gripper_state == 0:
                    self.gripper_vec = self.gripper_dict["open"]
                    self.gripper_blocked = False
                elif grasp >= -self.grasp_threshold and self.gripper_state == 1:
                    self.gripper_vec = self.gripper_dict["closed"]
                    self.gripper_blocked = False
                elif grasp < -self.grasp_threshold and self.gripper_state == 1:
                    self.data.ctrl[self._gripper_ctrl_id] = self._GRIPPER_MAX
                    self.gripper_state = 0
                    self.gripper_vec = self.gripper_dict["opening"]
                    self.prev_grasp_time = self.data.time
                    self.prev_grasp = grasp
                    self.gripper_blocked = True
                    moving_gripper = True
                    target_sim_time = self.data.time + self.gripper_sleep
                elif grasp > self.grasp_threshold and self.gripper_state == 0:
                    self.data.ctrl[self._gripper_ctrl_id] = 0
                    self.gripper_state = 1
                    self.gripper_vec = self.gripper_dict["closing"]
                    self.prev_grasp_time = self.data.time
                    self.prev_grasp = grasp
                    self.gripper_blocked = True
                    moving_gripper = True
                    target_sim_time = self.data.time + self.gripper_sleep
        else:  # Continuous gripper control
            grasp_action = action[-1]

            gripper_speed = 0.005
            prev_grasp_action = self.prev_action[-1]

            current_gripper_pos = self.data.qpos[self._gripper_ctrl_id]
            new_target_pos = current_gripper_pos + -grasp_action * gripper_speed
            self.data.ctrl[self._gripper_ctrl_id] = np.clip(new_target_pos, 0.0, self._GRIPPER_MAX)
        
        return moving_gripper, target_sim_time
    
    def _get_vel(self):
        """
        Compute the Cartesian speed (linear and angular velocity) of the end-effector.
        
        Returns:
            cartesian_speed: A (6,) numpy array where the first 3 elements are the
                            linear velocities and the last 3 elements are the angular velocities.
        """
        dq = self.data.qvel[self._panda_dof_ids]
        J_v = np.zeros((3, self.model.nv), dtype=np.float64)
        J_w = np.zeros((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jacSite(self.model, self.data, J_v, J_w, self._pinch_site_id)
        J_v, J_w = J_v[:, self._panda_dof_ids], J_w[:, self._panda_dof_ids]
        J = np.vstack((J_v, J_w))
        dx = J @ dq
        return dx.astype(np.float32)
    


    def _get_obs(self):
        obs = {"state": {}}
        
        # --- TCP pose and velocity ---
        tcp_world_pos = self.data.sensor("pinch_pos").data.copy()
        # Ensure quaternion is in xyzw order for Rotation, then back to wxyz if needed by convention elsewhere
        # MuJoCo sensors output wxyz, np.roll(q, -1) makes it xyzw
        tcp_world_quat_xyzw = np.roll(self.data.sensor("pinch_quat").data, -1).copy() 
        
        if self.randomize_domain:
            # Noise for position
            # Use a default from cfg or a fallback value
            position_noise_std = self.cfg.get("domain_randomization", {}).get("ee_pos_noise_std", 0.005) 
            tcp_world_pos += self.np_random.normal(0, position_noise_std, size=3)
            
            # Noise for orientation
            orientation_noise_std = self.cfg.get("domain_randomization", {}).get("ee_ori_noise_std", 0.005)
            orientation_noise_axis_angle = self.np_random.normal(0, orientation_noise_std, size=3)
            small_rotation = Rotation.from_rotvec(orientation_noise_axis_angle)
            current_rotation = Rotation.from_quat(tcp_world_quat_xyzw) # Expects xyzw
            new_rotation = small_rotation * current_rotation
            tcp_world_quat_xyzw = new_rotation.as_quat() # Returns xyzw, normalized
        
        # Storing tcp_pose as [pos (3), quat_xyzw (4)]
        obs["state"]["tcp_pose"] = np.concatenate([tcp_world_pos, tcp_world_quat_xyzw]).astype(np.float32)
        obs["state"]["tcp_vel"] = self._get_vel() 
        obs["state"]["gripper_pos"] = np.array([2 * self.data.qpos[8] / self._GRIPPER_HOME[0]], dtype=np.float32)
        if self.discrete_gripper:
            obs["state"]["gripper_vec"] = self.gripper_vec.astype(np.float32)

        # --- Image observations ---
        if self.image_obs:
            obs["images"] = {}
            for cam_name in self.cameras:
                obs["images"][cam_name] = self._viewers[cam_name].render(render_mode="rgb_array")

        if not self.image_obs:
            strawberry_state_obs = self._get_strawberry_state_obs(tcp_world_pos)
            obs["state"].update(strawberry_state_obs)
        
        if self.render_mode == "human":
            self._viewers['wrist1'].render(self.render_mode)

        return obs

    
    def _compute_reward(self, action):
        reward = 0.0
        info = {}
        info['success'] = False
        return reward, info