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
import gc
from fruit_gym.envs.randomization import (
    lighting_noise,
    action_scale_noise,
    initial_state_noise,
    camera_noise,
    floor_noise,
    skybox_noise,
)

def load_config(config_path):
    with open(config_path, 'r') as config_file:
        return yaml.safe_load(config_file)

class PickMultiStrawbEnv(MujocoEnv, utils.EzPickle):
    r"""
    ## Description

    **PickMultiStrawbEnv** is a robotic manipulation environment in which a Franka Panda robot must reach for and grasp multiple red strawberries among 
    multiple distractor (green) strawberries. The environment supports domain randomization, image observations, and multi-target grasping. Upon a successful grasp 
    of a red strawberry, the target is removed (by making its associated geometries invisible) from the environment and the reward function is updated accordingly. 
    This environment is designed to challenge reinforcement learning algorithms in continuous control and multi-object manipulation tasks.

    ## Action Space

    The action space is continuous and is defined as a `Box` with shape `(ee_dof + 1,)` of type `float32`. For example, if `ee_dof=6`, then the action space is: Box(-1, 1, (7,), float32)


    An action is represented as:  
    `[z, y, x, roll, pitch, yaw, grasp], in the end effector frame. z and x swapped for intuitive first person control`

    | Num | Action Component   | Description                                                                                                   | Range   |
    |-----|--------------------|---------------------------------------------------------------------------------------------------------------|---------|
    | 0   | dz                  | Displacement along the z-axis (vertical movement)                                                             | [-1, 1] |
    | 1   | dy                  | Displacement along the y-axis                                                                                 | [-1, 1] |
    | 2   | dx                  | Displacement along the x-axis                                                                                 | [-1, 1] |
    | 3   | droll               | Rotation about the x-axis (roll)                                                                              | [-1, 1] |
    | 4   | dpitch              | Rotation about the y-axis (pitch)                                                                             | [-1, 1] |
    | 5   | dyaw                | Rotation about the z-axis (yaw)                                                                               | [-1, 1] |
    | 6   | dgrasp              | Grasp command. Values above a threshold indicate an attempt to grasp; below indicate release.                 | [-1, 1] |

    ## Observation Space

    The observation space is a dictionary containing both a state vector and (optionally) image observations.

    ### State

    The state is a dictionary with the following keys:

    - **tcp_pose (7 elements):**  
    The position (3 elements) and orientation (quaternion, 4 elements) of the robot’s end-effector.
    
    - **tcp_vel (6 elements):**  
    The linear and angular velocities of the end-effector.
    
    - **gripper_pos (1 element):**  
    The current opening of the gripper.
    
    - **gripper_vec (4 elements):**  
    A vector representing the gripper state (e.g., one-hot encoding of open, closed, etc.).

    Thus, the overall state vector has 7 + 6 + 1 + 4 = 18 elements.

    ### Images

    If `image_obs=True`, the observation also includes an `images` dictionary mapping camera names (e.g., `"wrist1"`, `"wrist2"`, `"front"`) to RGB images of 
    shape `(height, width, 3)` with values in `[0, 255]`.

    ## Rewards

    The total reward is a weighted sum of several components:

    - **r_red:** A positive reward proportional to how close the end-effector is to the nearest red strawberry target.
    - **r_green:** A penalty based on the movement of green (distractor) objects from their initial positions.
    - **r_grasp:** A binary reward given when both gripper fingers make contact with a desired strawberry stem.
    - **r_energy:** A penalty proportional to the magnitude of the action, encouraging energy-efficient control.
    - **r_smooth:** A penalty for large changes in actions between consecutive steps to promote smooth control.
    - **r_bad_grasp:** A penalty for grasping the wrong things.

    When a successful grasp is detected (both fingers contact a target stem), the corresponding strawberry is removed from the environment by making its associated geometries invisible 
    (setting their collision parameters to 0), and its index is removed from the active target lists.

    ## Starting State

    At the beginning of each episode, the robot arm and gripper are reset to their home positions. The positions and orientations of the target vine (holding the strawberries) 
    and the distractor vines are randomized within predefined bounds. Domain randomization may also be applied to lighting, camera parameters, and object properties as specified in a configuration file.

    ## Episode End

    ### Termination

    An episode terminates when either all red strawberry targets have been grasped (i.e., removed from the environment) or an external time limit is reached.

    ### Truncation

    Truncation is managed by the Gymnasium `TimeLimit` wrapper, and is not handled intrinsically by the environment.

    ## Arguments

    The environment accepts a variety of parameters upon instantiation:
    | Parameter               | Type       | Default                        | Description                                                                                       |
    |----------------------   |------------|--------------------------------|---------------------------------------------------------------------------------------------------|
    | `image_obs`             | bool       | True                           | Whether to include image observations.                                                            |
    | `include_privileged_obs`| bool       | False                          | Whether to include privileged observations (e.g., object states not visible to the agent).        |
    | `randomize_domain`      | bool       | True                           | Whether to apply domain randomization to lighting, camera, and object properties.                 |
    | `ee_dof`                | int        | 6                              | Degrees of freedom for the end-effector (3 for position only; 6 for position and orientation).    |
    | `control_dt`            | float      | 0.05                           | Time interval between control updates.                                                            |
    | `physics_dt`            | float      | 0.002                          | Simulation time step.                                                                             |
    | `width`                 | int        | 480                            | Image width (if `image_obs` is True).                                                             |
    | `height`                | int        | 480                            | Image height (if `image_obs` is True).                                                            |
    | `pos_scale`             | float      | 0.008                          | Scaling factor for positional changes.                                                            |
    | `rot_scale`             | float      | 0.5                            | Scaling factor for rotational changes.                                                            |
    | `cameras`               | List[str]  | ["wrist1", "wrist2", "front"]  | List of camera names for rendering images.                                                        |
    | `reward_type`           | str        | "dense"                        | Reward type; can be "dense" or "sparse".                                                          |
    | `gripper_pause`         | bool       | False                          | If True, the simulation pauses briefly after a gripper action.                                    |
    | `render_mode`           | str        | "rgb_array"                    | Rendering mode, e.g., "human" or "rgb_array".                                                  
    """


    metadata = { 
        "render_modes": ["human", "rgb_array", "depth_array"], 
    }
    
    def __init__(
        self,
        image_obs: bool = True,
        include_privileged_obs: bool = False,
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
        disappear_delay_steps: int = 16,
        render_mode: str = "rgb_array",
        config_path: Optional[Union[str, Path]] = None,
        **kwargs,
    ):
        utils.EzPickle.__init__(self, image_obs=image_obs, **kwargs)

        if cameras is None:
            cameras = ["wrist1", "wrist2", "front"]

        self.image_obs = image_obs
        self.include_privileged_obs = include_privileged_obs
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
        self.disappear_delay_steps = disappear_delay_steps
        self._pending_removals = {}   # {strawb_idx: steps_left}
        self._grasped_pending = set() # indices already credited with r_grasp, awaiting disappearance

        self._PANDA_HOME = np.array([0.0, -1.6, 0.0, -2.54, -0.05, 2.49, 0.822], dtype=np.float32)
        self._GRIPPER_HOME = np.array([0.0141, 0.0141], dtype=np.float32)
        self._GRIPPER_MIN = 0
        self._GRIPPER_MAX = 0.004
        self._PANDA_XYZ = np.array([0.1, 0, 0.8], dtype=np.float32)
        self._CARTESIAN_BOUNDS = np.array([[0.05, -0.2, 0.6], [0.55, 0.2, 0.95]], dtype=np.float32)
        self._ROTATION_BOUNDS = np.array([[-np.pi/3, -np.pi/6, -np.pi/10],[np.pi/3, np.pi/6, np.pi/10]], dtype=np.float32)
        self.default_obj_pos = np.array([0.42, 0, 0.85])
        self._blocks_picked = 0
        self.gripper_sleep = 0.2
        self.grasp_threshold = 0.333
        MAX_OBSERVABLE_STRAWBERRIES = 8

        if config_path is None:
            config_path = Path(__file__).parent.parent / "configs" / "strawb_hanging.yaml"
        self.cfg = load_config(config_path)

        # Define the basic state space components
        state_space_dict = {
            "tcp_pose": Box(-np.inf, np.inf, shape=(7,), dtype=np.float32), # 3 pos, 4 quat (xyzw)
            "tcp_vel": Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
            "gripper_pos": Box(-1, 1, shape=(1,), dtype=np.float32),
        }
        if self.discrete_gripper:
            state_space_dict["gripper_vec"] = Box(0, 1, shape=(4,), dtype=np.float32)

        if not image_obs:
            # Add detailed strawberry information if not using image observations
            state_space_dict["all_red_pos_relative"] = Box(
                -np.inf, np.inf, shape=(MAX_OBSERVABLE_STRAWBERRIES, 3), dtype=np.float32
            )
            state_space_dict["all_red_distances"] = Box(
                0, np.inf, shape=(MAX_OBSERVABLE_STRAWBERRIES,), dtype=np.float32
            )
            state_space_dict["all_red_mask"] = Box(
                0.0, 1.0, shape=(MAX_OBSERVABLE_STRAWBERRIES,), dtype=np.float32
            )
            state_space_dict["all_green_pos_relative"] = Box(
                -np.inf, np.inf, shape=(MAX_OBSERVABLE_STRAWBERRIES, 3), dtype=np.float32
            )
            state_space_dict["all_green_distances"] = Box(
                0, np.inf, shape=(MAX_OBSERVABLE_STRAWBERRIES,), dtype=np.float32
            )
            state_space_dict["all_green_mask"] = Box(
                0.0, 1.0, shape=(MAX_OBSERVABLE_STRAWBERRIES,), dtype=np.float32
            )
            state_space_dict["num_red_left"] = Box(
                0, MAX_OBSERVABLE_STRAWBERRIES, shape=(1,), dtype=np.float32
            )

        # Define privileged observation space separately for asymmetric RL
        priv_state_space_dict = {}
        if include_privileged_obs:
            # Distance and alignment information
            priv_state_space_dict["min_red_distance"] = Box(
                0, np.inf, shape=(1,), dtype=np.float32
            )
            priv_state_space_dict["gripper_alignment_quality"] = Box(
                0, np.inf, shape=(1,), dtype=np.float32  # radial distance from gripper axis
            )
            # Grasp quality indicators
            priv_state_space_dict["good_grasp_detected"] = Box(
                0.0, 1.0, shape=(1,), dtype=np.float32
            )
            priv_state_space_dict["bad_grasp_detected"] = Box(
                0.0, 1.0, shape=(1,), dtype=np.float32
            )
            # Collision and positioning
            priv_state_space_dict["collision_detected"] = Box(
                0.0, 1.0, shape=(1,), dtype=np.float32
            )
            priv_state_space_dict["red_stems_in_box_count"] = Box(
                0, MAX_OBSERVABLE_STRAWBERRIES, shape=(1,), dtype=np.float32
            )
            priv_state_space_dict["green_stems_in_box_count"] = Box(
                0, MAX_OBSERVABLE_STRAWBERRIES, shape=(1,), dtype=np.float32
            )
            # Contact information for each finger
            priv_state_space_dict["left_finger_contacts"] = Box(
                0, 10, shape=(1,), dtype=np.float32  # Number of contacts
            )
            priv_state_space_dict["right_finger_contacts"] = Box(
                0, 10, shape=(1,), dtype=np.float32
            )
            # Additional useful metrics
            priv_state_space_dict["total_distractor_displacement"] = Box(
                0, np.inf, shape=(1,), dtype=np.float32
            )

        self.observation_space = Dict({"state": Dict(state_space_dict)})
        if include_privileged_obs:
            self.observation_space["priv_state"] = Dict(priv_state_space_dict)
        if image_obs:
            self.observation_space["images"] = Dict()
            for camera in self.cameras:
                self.observation_space["images"][camera] = Box(
                    0, 255, shape=(self.height, self.width, 3), dtype=np.uint8
                )

        p = Path(__file__).parent
        env_dir = os.path.join(p, "xmls/mjmodel.xml")
        self._n_substeps = int(float(control_dt) / float(physics_dt))
        self.frame_skip = 1

        MujocoEnv.__init__(
            self, 
            env_dir, 
            self.frame_skip, 
            observation_space=self.observation_space, 
            render_mode=self.render_mode,
            width=self.width,
            height=self.height, 
            camera_id=0, 
            **kwargs,
        )
        self.model.opt.timestep = physics_dt
        self.camera_id = ()
        for cam in self.cameras:
            self.camera_id += (self.model.camera(cam).id,)
        self.action_space = Box(
            np.array([-1.0]*(self.ee_dof+1)), 
            np.array([1.0]*(self.ee_dof+1)),
            dtype=np.float32,
        )
        self._viewer = MujocoRenderer(self.model, self.data,)
        self.setup()

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

        for i in range(self.model.ntex):
            if i < self.model.ntex - 1:
                # For all but the last texture, use the next index
                name_start = self.model.name_texadr[i]
                name_end = self.model.name_texadr[i + 1] - 1
            else:
                # For the last texture, go until the first null byte or the end of the names array
                name_start = self.model.name_texadr[i]
                name_end = len(self.model.names)
            # Decode the name slice
            texture_name = self.model.names[name_start:name_end].split(b'\x00', 1)[0].decode('utf-8')
            if self.model.texture(texture_name).type[0] == 2:
                self.skybox_tex_ids.append(self.model.texture(texture_name).id)
            else:
                self.floor_tex_ids.append(self.model.texture(texture_name).id)
        self.initial_vine_rotation = Rotation.from_quat(np.roll(self.model.body_quat[self.model.body("vine1").id], -1))

        self.initial_position = np.array([0.1, 0.0, 0.75], dtype=np.float32)
        self.initial_orientation = [0.725, 0.0, 0.688, 0.0]
        self.initial_rotation = Rotation.from_quat(self.initial_orientation)

        self.init_headlight_diffuse = self.model.vis.headlight.diffuse.copy()
        self.init_headlight_ambient = self.model.vis.headlight.ambient.copy()
        self.init_headlight_specular = self.model.vis.headlight.specular.copy()

        self.num_green = 7
        self.model.body_pos[self.model.body("vine1").id] = self.default_obj_pos
        for i in range(2, self.num_green+2):
            self.model.body_pos[self.model.body(f"vine{i}").id] = self.default_obj_pos + np.array([-0.05, 0.0, 0.0])
        self.active_indices = np.array(list(range(2, self.num_green + 2)))

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

    def object_noise(self):
        dr = self.cfg.get("domain_randomization", {})
        object_cfg = dr.get("objects", {})
        if not object_cfg.get("enabled", False):
            return
        
        # --- Store initial geom properties once, then restore them each call ---
        if not hasattr(self, '_initial_geom_properties_stored'):
            self._initial_geom_rgba = self.model.geom_rgba.copy()
            self._initial_geom_contype = self.model.geom_contype.copy()
            self._initial_geom_conaffinity = self.model.geom_conaffinity.copy()
            self._initial_geom_group = self.model.geom_group.copy() # Store initial groups
            self._initial_geom_properties_stored = True
        
        # Restore all geoms to their original XML-defined state
        self.model.geom_rgba[:] = self._initial_geom_rgba
        self.model.geom_contype[:] = self._initial_geom_contype
        self.model.geom_conaffinity[:] = self._initial_geom_conaffinity
        self.model.geom_group[:] = self._initial_geom_group # Restore initial groups

        # Target pos
        target_pos_noise_low = object_cfg.get("target_pos_noise_low", [0.0, 0.0, 0.0])
        target_pos_noise_high = object_cfg.get("target_pos_noise_high", [0.0, 0.0, 0.0])
        target_pos_noise = np.random.uniform(low=target_pos_noise_low, high=target_pos_noise_high, size=3)
        target_pos = self.data.sensor("pinch_pos").data.copy()
        target_pos[0] += 0.15
        target_pos[2] += 0.2
        self.model.body_pos[self.model.body("vine1").id] = target_pos
        # Target orientation
        random_z_angle = np.random.uniform(low=-np.pi, high=np.pi) # Random angle in radians
        z_rotation = Rotation.from_euler('z', random_z_angle)
        new_rotation = z_rotation * self.initial_vine_rotation
        new_quat = new_rotation.as_quat()
        self.model.body_quat[self.model.body("vine1").id] = [new_quat[3], new_quat[0], new_quat[1], new_quat[2]]

        red_rgba = np.array([0.55, 0.1, 0.1, 1])
        green_rgba = np.array([0.5, 0.63, 0.45, 1])
        
        # `vine1` is always the primary target and is always red.
        self.red_blocks = [1] 
        self.green_blocks = []
        self.red_positions = {}
        self.green_positions = {}

        
        target_names = ["block1", "block1_big", "block1_small"]
        sub_geom_ids = {}
        for name in target_names:
            sub_body = self.model.body(name)
            geom_start = self.model.body_geomadr[sub_body.id]
            geom_count = self.model.body_geomnum[sub_body.id]
            sub_geom_ids[name] = list(range(geom_start, geom_start + geom_count))

        active_sub = np.random.choice(target_names)
        for name in target_names:
            for geom_id in sub_geom_ids[name]:
                geom_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
                if name == active_sub:
                    if geom_name == active_sub:
                        active_geom_name = geom_name
                        self.model.geom_group[geom_id] = 3
                        self.model.geom_contype[geom_id] = 1
                        self.model.geom_conaffinity[geom_id] = 1
                    else:
                        self.model.geom_group[geom_id] = 0
                        self.model.geom_contype[geom_id] = 0
                        self.model.geom_conaffinity[geom_id] = 0
                else:
                    self.model.geom_group[geom_id] = 3
                    self.model.geom_contype[geom_id] = 0
                    self.model.geom_conaffinity[geom_id] = 0

        distract_pos_noise_low = object_cfg.get("distract_pos_noise_low", [0.0, 0.0, 0.0])
        distract_pos_noise_high = object_cfg.get("distract_pos_noise_high", [0.0, 0.0, 0.0])

        distractor_indices = list(range(2, self.num_green + 2))
        active_count = np.random.randint(4, len(distractor_indices) + 1)
        active_indices = np.random.choice(distractor_indices, size=active_count, replace=False)
        self.active_indices = active_indices
        self.active_visual_geoms = {}
        
        # 1. Define the desired TOTAL number of red strawberries.
        min_total_red = 2
        max_total_red_from_config = object_cfg.get("max_red_strawberries", 3)
        
        # 2. Since block1 is always red, we need to select (N-1) from the distractors.
        #    The number of distractors to make red is in the range [min-1, max-1].
        min_distractors_to_make_red = max(0, min_total_red - 1) # e.g., 2-1=1
        max_distractors_to_make_red = max(0, max_total_red_from_config - 1) # e.g., 4-1=3
        
        # 3. Ensure the number to make doesn't exceed available active distractors.
        effective_max = min(max_distractors_to_make_red, len(active_indices))
        
        # 4. Ensure the minimum is not greater than this effective max.
        effective_min = min(min_distractors_to_make_red, effective_max)
        
        # 5. Determine how many distractors to color red for this episode.
        num_distractors_to_make_red = 0
        if effective_min <= effective_max:
            num_distractors_to_make_red = np.random.randint(effective_min, effective_max + 1)
        
        # 6. Select the candidates from the active pool.
        if num_distractors_to_make_red > 0:
            red_candidate_indices = np.random.choice(active_indices, size=num_distractors_to_make_red, replace=False)
        else:
            red_candidate_indices = []

        for i in distractor_indices:
            vine_body_name = f"vine{i}"
            vine_body_id = self.model.body(vine_body_name).id
            # Randomize the distractor vine's position.
            distract_pos_noise = np.random.uniform(low=distract_pos_noise_low,
                                                high=distract_pos_noise_high,
                                                size=3)
            vine_body = self.model.body(f"vine{i}")
            self.model.body_pos[vine_body.id] = target_pos + distract_pos_noise

            # Randomize its orientation.
            random_z_angle = np.random.uniform(low=-np.pi, high=np.pi)
            z_rotation = Rotation.from_euler('z', random_z_angle)
            new_rotation = z_rotation * self.initial_vine_rotation
            new_quat = new_rotation.as_quat()
            self.model.body_quat[vine_body.id] = [new_quat[3], new_quat[0], new_quat[1], new_quat[2]]

            # change strawb size
            sub_names = [f"block{i}", f"block{i}_big", f"block{i}_small"]
            sub_geom_ids = {}
            # Gather geom id lists for each sub-body.
            for name in sub_names:
                sub_body = self.model.body(name)
                geom_start = self.model.body_geomadr[sub_body.id]
                geom_count = self.model.body_geomnum[sub_body.id]
                sub_geom_ids[name] = list(range(geom_start, geom_start + geom_count))

            # If this vine is NOT active, disable its collisions.
            if i not in active_indices:
                if object_cfg.get("hide_inactive_vines", True):
                    self._set_inactive_properties_recursive(vine_body_id)
                else:
                    for name in sub_names:
                        for geom_id in sub_geom_ids[name]:
                            self.model.geom_group[geom_id] = 3
                            self.model.geom_contype[geom_id] = 0
                            self.model.geom_conaffinity[geom_id] = 0
            else:
                # Assign color based on whether the index was chosen to be red
                if i in red_candidate_indices:
                    chosen_rgba = red_rgba
                    colour = "red"
                else:
                    chosen_rgba = green_rgba
                    colour = "green"
                    
                active_sub = np.random.choice(sub_names)
                for name in sub_names:
                    for geom_id in sub_geom_ids[name]:
                        geom_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
                        if name == active_sub:
                            if geom_name == f"{name}_visual":
                                self.model.geom_rgba[geom_id] = chosen_rgba
                                self.active_visual_geoms[i] = geom_id
                                # This is where the lists are populated for the rest of the sim
                                if colour == "red":
                                    self.red_blocks.append(i)
                                elif colour == "green":
                                    self.green_blocks.append(i)                               
                            if geom_name == name:
                                self.model.geom_group[geom_id] = 3
                                self.model.geom_contype[geom_id] = 1
                                self.model.geom_conaffinity[geom_id] = 1
                            else:
                                self.model.geom_group[geom_id] = 0
                                self.model.geom_contype[geom_id] = 0
                                self.model.geom_conaffinity[geom_id] = 0
                        else:
                            self.model.geom_group[geom_id] = 3
                            self.model.geom_contype[geom_id] = 0
                            self.model.geom_conaffinity[geom_id] = 0

        self.data.qvel[:] = 0
        self.data.qacc[:] = 0
        mujoco.mj_forward(self.model, self.data)
        for i in self.red_blocks:
            self.red_positions[i] = self.data.sensor(f"block{i}_pos").data.copy()
        for j in self.green_blocks:
            self.green_positions[j] = self.data.sensor(f"block{j}_pos").data.copy()


    def domain_randomization(self) -> None:
        dr = self.cfg.get("domain_randomization", {})
        if dr.get("objects", {}).get("enabled", False):
            self.object_noise()
        if dr.get("lighting", {}).get("enabled", False):
            lighting_noise(self)
        if dr.get("action_scale", {}).get("enabled", False):
            action_scale_noise(self)
        if dr.get("initial_state", {}).get("enabled", False):
            initial_state_noise(self)
        if dr.get("cameras", {}).get("enabled", False):
            camera_noise(self)
        if dr.get("skybox", {}).get("enabled", False):
            skybox_noise(self)
        if dr.get("floor", {}).get("enabled", False):
            floor_noise(self)
        self._viewer = MujocoRenderer(self.model, self.data)

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
            self.reset_arm_and_gripper()
            if self.randomize_domain:
                self.domain_randomization()

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
            
            self._block_init = self.data.sensor("block1_pos").data
            self._x_success = self._block_init[0] - 0.1
            self._z_success = self._block_init[2] + 0.05
            self._block_success = self._block_init.copy()
            self._block_success[0] = self._x_success
            self._block_success[2] = self._z_success
            self._blocks_picked = 0
            self._pending_removals = {}
            self._grasped_pending = set()

            for i in self.red_blocks:
                self.red_positions[i] = self.data.sensor(f"block{i}_pos").data.copy()
            for j in self.green_blocks:
                self.green_positions[j] = self.data.sensor(f"block{j}_pos").data.copy()

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
        self.current_privileged_info = self._compute_privileged_info()
        obs = self._get_obs()
        if self.render_mode == "human":
            self.render()

        # Reward
        reward, info = self._compute_reward(action)

        # Disappear picked strawberries
        self._tick_removal_timers()

        if self.reward_type == "sparse":
            info['dense_reward'] = reward
            if info['r_grasp'] > 0:
                reward = 1.0
            else:
                reward = 0.0
        if info['success'] == True:
            terminated = True
        else:
            terminated = False
        self.prev_gripper_state = self.gripper_state

        return obs, reward, terminated, False, info 
    
    def render(self):
        rendered_frames = []
        for cam_id in self.camera_id:
            rendered_frames.append(
                self._viewer.render(render_mode="rgb_array", camera_id=cam_id)
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
    
    def _get_strawberry_state_obs(self, tcp_world_pos):
        """
        Compute state-based strawberry observations (positions, distances, masks).
        This includes both red and green strawberry information when not using image observations.
        """
        MAX_OBSERVABLE_STRAWBERRIES = self.num_green + 1 
        
        # Use the (potentially noised) tcp_world_pos for relative calculations
        tcp_current_pos_for_relative = tcp_world_pos 

        # --- Red Strawberry STEMS: Relative Positions, Distances, and Mask ---
        red_stem_positions_relative_sorted = np.zeros((MAX_OBSERVABLE_STRAWBERRIES, 3), dtype=np.float32)
        red_stem_distances_sorted = np.zeros(MAX_OBSERVABLE_STRAWBERRIES, dtype=np.float32)
        red_stem_mask_sorted = np.zeros(MAX_OBSERVABLE_STRAWBERRIES, dtype=np.float32)
        
        active_red_stem_data = []
        if hasattr(self, 'red_blocks') and self.red_blocks:
            for block_idx in self.red_blocks:
                try:
                    strawberry_stem_pos_world = self.data.sensor(f"stem{block_idx}_pos").data.copy()
                    dist = np.linalg.norm(strawberry_stem_pos_world - tcp_current_pos_for_relative)
                    relative_pos = strawberry_stem_pos_world - tcp_current_pos_for_relative
                    active_red_stem_data.append({'distance': dist, 'relative_pos': relative_pos})
                except Exception as e:
                    # print(f"Warning: Could not get stem pos for red_block {block_idx}: {e}")
                    pass
        
        active_red_stem_data.sort(key=lambda s: s['distance'])

        for i, data in enumerate(active_red_stem_data):
            if i < MAX_OBSERVABLE_STRAWBERRIES:
                red_stem_positions_relative_sorted[i, :] = data['relative_pos']
                red_stem_distances_sorted[i] = data['distance']
                red_stem_mask_sorted[i] = 1.0 
            else:
                break 

        # --- Green Strawberry BLOCKS: Relative Positions, Distances, and Mask ---
        green_block_positions_relative_sorted = np.zeros((MAX_OBSERVABLE_STRAWBERRIES, 3), dtype=np.float32)
        green_block_distances_sorted = np.zeros(MAX_OBSERVABLE_STRAWBERRIES, dtype=np.float32)
        green_block_mask_sorted = np.zeros(MAX_OBSERVABLE_STRAWBERRIES, dtype=np.float32)

        active_green_block_data = []
        if hasattr(self, 'green_blocks') and self.green_blocks: # Ensure self.green_blocks is populated
            for block_idx in self.green_blocks:
                try:
                    strawberry_block_pos_world = self.data.sensor(f"block{block_idx}_pos").data.copy()
                    dist = np.linalg.norm(strawberry_block_pos_world - tcp_current_pos_for_relative)
                    relative_pos = strawberry_block_pos_world - tcp_current_pos_for_relative
                    active_green_block_data.append({'distance': dist, 'relative_pos': relative_pos})
                except Exception as e:
                    # print(f"Warning: Could not get block pos for green_block {block_idx}: {e}")
                    pass
        
        active_green_block_data.sort(key=lambda s: s['distance'])

        for i, data in enumerate(active_green_block_data):
            if i < MAX_OBSERVABLE_STRAWBERRIES:
                green_block_positions_relative_sorted[i, :] = data['relative_pos']
                green_block_distances_sorted[i] = data['distance']
                green_block_mask_sorted[i] = 1.0
            else:
                break
        
        # --- Number of remaining red strawberries ---
        num_red_left = 0
        if hasattr(self, 'red_blocks'):
            num_red_left = len(self.red_blocks)
        
        # Return dictionary of strawberry state observations
        return {
            "all_red_pos_relative": red_stem_positions_relative_sorted,
            "all_red_distances": red_stem_distances_sorted,
            "all_red_mask": red_stem_mask_sorted,
            "all_green_pos_relative": green_block_positions_relative_sorted,
            "all_green_distances": green_block_distances_sorted,
            "all_green_mask": green_block_mask_sorted,
            "num_red_left": np.array([num_red_left], dtype=np.float32)
        }

    def _compute_privileged_info(self):
        """
        Compute privileged information for asymmetric RL.
        
        Returns privileged observations that are useful for the critic but not
        available to the actor, including grasp quality, collision state, and
        precise distance/alignment measurements.
        
        Returns:
            dict: Dictionary containing privileged information with keys:
                - min_red_dist: Distance to nearest red strawberry
                - radial_dist: Alignment quality measure
                - good_grasp: Whether a good grasp is detected
                - bad_grasp: Whether a bad grasp is detected
                - collision_detected: Whether unwanted collisions occurred
                - stem_in_box: Whether a red stem is in the gripper box
                - left_finger_contacts: Number of left finger contacts
                - right_finger_contacts: Number of right finger contacts
                - total_displacement: Total displacement of distractor objects
                - grasped_idx: Index of grasped strawberry (if any)
        """

        tcp_pos = self.data.sensor("long_pinch_pos").data
        left_pinch_pos = self.data.sensor("left_pinch_pos").data
        right_pinch_pos = self.data.sensor("right_pinch_pos").data
        
        # Initialize return dict
        info = {
            "min_red_dist": float('inf'),
            "radial_dist": 0.0,
            "good_grasp": False,
            "bad_grasp": False,
            "collision_detected": False,
            "red_stems_in_box_count": 0,
            "green_stems_in_box_count": 0,
            "left_finger_contacts": 0,
            "right_finger_contacts": 0,
            "total_displacement": 0.0,
            "grasped_idx": None  # Add this to track which strawberry was grasped
        }
        
        # Distance to nearest red strawberry
        if len(self.red_blocks) > 0:
            dists = {}
            for red_idx in self.red_blocks:
                stem_pos = self.data.sensor(f"stem{red_idx}_pos").data
                dists[red_idx] = (np.linalg.norm(stem_pos - left_pinch_pos) + 
                                np.linalg.norm(stem_pos - right_pinch_pos)) / 2.0
            
            if dists:
                closest_red_idx = min(dists, key=dists.get)
                info["min_red_dist"] = dists[closest_red_idx]
                closest_red_stem_pos = self.data.sensor(f"stem{closest_red_idx}_pos").data
                
                # Compute alignment quality
                pinch_rot_mat = self.data.site_xmat[self._pinch_site_id].reshape(3, 3)
                gripper_y_axis = pinch_rot_mat[:, 1]
                vec_to_stem = closest_red_stem_pos - tcp_pos
                proj_y = np.dot(vec_to_stem, gripper_y_axis)
                info["radial_dist"] = abs(proj_y)
                
         # Check how many RED stems are in the gripper box
        if hasattr(self, 'red_blocks'):
            for red_idx in self.red_blocks:
                try:
                    stem_pos = self.data.sensor(f"stem{red_idx}_pos").data
                    if self.is_stem_in_gripper_box(stem_pos):
                        info["red_stems_in_box_count"] += 1
                except Exception:
                    pass

        # Check how many GREEN stems are in the gripper box
        green_vine_part_in_box = False
        if hasattr(self, 'green_blocks'):
            for green_idx in self.green_blocks:
                try:
                    stem_pos = self.data.sensor(f"stem{green_idx}_pos").data
                    if self.is_stem_in_gripper_box(stem_pos):
                        green_vine_part_in_box = True
                        break  # A green vine is in the box, no need to check other green vines
                except Exception:
                    pass
                # Determine the character prefix for the vine's geoms (b for 2, c for 3, etc.)
                prefix_char = chr(ord('`') + green_idx)
                # Loop through the 4 geoms that make up the vine (e.g., bG0, bG1, bG2, bG3)
                for i in range(4):
                    geom_name = f"{prefix_char}G{i}"
                    try:
                        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
                        geom_pos = self.data.geom_xpos[geom_id]
                        if self.is_stem_in_gripper_box(geom_pos):
                            green_vine_part_in_box = True
                            break  # A part is in the box, no need to check other parts of this vine
                    except KeyError:
                        # This geom name doesn't exist, which is fine. Continue to the next.
                        pass
                if green_vine_part_in_box:
                    break # A green vine is in the box, no need to check other green vines

        # Update the info dict based on our findings. This re-uses the existing reward logic.
        info["green_stems_in_box_count"] = 1 if green_vine_part_in_box else 0
        
        # Contact analysis - track which stems each finger contacts
        left_contacts = 0
        right_contacts = 0
        collision_detected = False
        left_finger_contact_good = False
        right_finger_contact_good = False
        left_finger_contact_bad = False
        right_finger_contact_bad = False
        grasped_idx = None
        
        allowed_prefixes = [f"{chr(ord('`')+i)}G3" for i in self.red_blocks]
        
        for i in range(self.data.ncon):
            geom1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, self.data.contact[i].geom1) or ""
            geom2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, self.data.contact[i].geom2) or ""
            
            # Check for unwanted collisions
            if ("finger" in geom1_name) or ("finger" in geom2_name):
                if ("block" in geom1_name) or ("block" in geom2_name):
                    collision_detected = True
            
            # Analyze left finger contacts
            if "left_finger_inner" in (geom1_name, geom2_name):
                left_contacts += 1
                other = geom1_name if geom2_name == "left_finger_inner" else geom2_name
                if other.startswith("stem"):
                    try:
                        stem_idx = int(other[len("stem"):])
                        if stem_idx in self.red_blocks:
                            left_finger_contact_good = True
                            grasped_idx = stem_idx
                        else:
                            left_finger_contact_bad = True
                    except ValueError:
                        pass
                elif other not in allowed_prefixes and other != "right_finger_inner":
                    left_finger_contact_bad = True
                    
            # Analyze right finger contacts
            if "right_finger_inner" in (geom1_name, geom2_name):
                right_contacts += 1
                other = geom1_name if geom2_name == "right_finger_inner" else geom2_name
                if other.startswith("stem"):
                    try:
                        stem_idx = int(other[len("stem"):])
                        if stem_idx in self.red_blocks:
                            right_finger_contact_good = True
                            # Only update grasped_idx if both fingers contact the same stem
                            if grasped_idx is None or grasped_idx == stem_idx:
                                grasped_idx = stem_idx
                            else:
                                # Different stems contacted by different fingers - no good grasp
                                grasped_idx = None
                        else:
                            # If the right finger contacts a green stem, it's a bad grasp
                            right_finger_contact_bad = True
                    except ValueError:
                        pass
                elif other not in allowed_prefixes and other != "left_finger_inner":
                    right_finger_contact_bad = True
        
        # Good grasp only if BOTH fingers contact good targets
        good_grasp = left_finger_contact_good and right_finger_contact_good
        bad_grasp = left_finger_contact_bad and right_finger_contact_bad
        
        info["left_finger_contacts"] = left_contacts
        info["right_finger_contacts"] = right_contacts
        info["collision_detected"] = collision_detected
        info["good_grasp"] = good_grasp
        info["bad_grasp"] = bad_grasp
        info["grasped_idx"] = grasped_idx if good_grasp else None
        
        # Total distractor displacement
        total_displacement = 0.0
        for j in self.green_blocks:
            try:
                current_pos = self.data.sensor(f"block{j}_pos").data
                initial_pos = self.green_positions[j]
                total_displacement += np.linalg.norm(current_pos - initial_pos)
            except KeyError:
                pass
        for i in self.red_blocks:
            try:
                current_pos = self.data.sensor(f"block{i}_pos").data
                initial_pos = self.red_positions[i]
                total_displacement += np.linalg.norm(current_pos - initial_pos)
            except KeyError:
                pass
        info["total_displacement"] = total_displacement
        
        return info

    def is_stem_in_gripper_box(self, stem_pos: np.ndarray) -> bool:
        """
        Checks if a world-space point is within the 3D box defined by two
        central gripper sites and constant height/depth values.
        """
        BOX_HEIGHT = 0.041
        BOX_DEPTH = 0.038  
        
        # 1. Get the world positions of the two central sites
        pos_left = self.data.site('left_pinch').xpos
        pos_right = self.data.site('right_pinch').xpos
        # 2. Define the box's center and orientation
        # The origin is the midpoint between the two fingers
        box_origin = (pos_left + pos_right) / 2
        # Use the main 'pinch' site for a stable orientation reference
        box_orientation = self.data.site('long_pinch').xmat.reshape(3, 3)
        # 3. Define the box's dimensions
        # Width is calculated live, height and depth are from constants
        box_width = np.linalg.norm(pos_left - pos_right)
        # 4. Transform the stem's position into the box's local coordinate frame
        vec_world = stem_pos - box_origin
        local_stem_pos = box_orientation.T @ vec_world
        # 5. Perform the checks in the simple, local coordinate system
        # Check height against the constant BOX_HEIGHT
        in_height = -BOX_HEIGHT / 2 <= local_stem_pos[0] <= BOX_HEIGHT / 2
        # Check width against the live-measured box_width
        in_width = -box_width / 2 <= local_stem_pos[1] <= box_width / 2
        # Check depth against the constant BOX_DEPTH. Assumes origin is in the middle of the depth.
        in_depth = -BOX_DEPTH / 2 <= local_stem_pos[2] <= BOX_DEPTH / 2

        return in_height and in_width and in_depth

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
            tcp_world_pos += np.random.normal(0, position_noise_std, size=3)
            
            # Noise for orientation
            orientation_noise_std = self.cfg.get("domain_randomization", {}).get("ee_ori_noise_std", 0.005)
            orientation_noise_axis_angle = np.random.normal(0, orientation_noise_std, size=3)
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

        if self.include_privileged_obs:
            privileged_info = getattr(self, 'current_privileged_info', self._compute_privileged_info())
            
            obs["priv_state"] = {}
            obs["priv_state"]["min_red_distance"] = np.array([privileged_info["min_red_dist"]], dtype=np.float32)
            obs["priv_state"]["gripper_alignment_quality"] = np.array([privileged_info["radial_dist"]], dtype=np.float32)
            obs["priv_state"]["good_grasp_detected"] = np.array([float(privileged_info["good_grasp"])], dtype=np.float32)
            obs["priv_state"]["bad_grasp_detected"] = np.array([float(privileged_info["bad_grasp"])], dtype=np.float32)
            obs["priv_state"]["collision_detected"] = np.array([float(privileged_info["collision_detected"])], dtype=np.float32)
            obs["priv_state"]["red_stems_in_box_count"] = np.array([privileged_info["red_stems_in_box_count"]], dtype=np.float32)
            obs["priv_state"]["green_stems_in_box_count"] = np.array([privileged_info["green_stems_in_box_count"]], dtype=np.float32)
            obs["priv_state"]["left_finger_contacts"] = np.array([privileged_info["left_finger_contacts"]], dtype=np.float32)
            obs["priv_state"]["right_finger_contacts"] = np.array([privileged_info["right_finger_contacts"]], dtype=np.float32)
            obs["priv_state"]["total_distractor_displacement"] = np.array([privileged_info["total_displacement"]], dtype=np.float32)

        # --- Image observations ---
        if self.image_obs:
            obs["images"] = {}
            for cam_name in self.cameras:
                cam_id = self.model.camera(cam_name).id
                obs["images"][cam_name] = self._viewer.render(render_mode="rgb_array", camera_id=cam_id)

        if not self.image_obs:
            strawberry_state_obs = self._get_strawberry_state_obs(tcp_world_pos)
            obs["state"].update(strawberry_state_obs)
        
        if self.render_mode == "human":
            self._viewer.render(self.render_mode)

        return obs

    
    def _compute_reward(self, action):
        # Use cached privileged information from step()
        privileged_info = getattr(self, 'current_privileged_info', self._compute_privileged_info())
        
        # Extract values for reward computation
        min_red_dist = privileged_info["min_red_dist"]
        good_grasp = privileged_info["good_grasp"]
        bad_grasp = privileged_info["bad_grasp"]
        collision_detected = privileged_info["collision_detected"]
        red_stems_in_box_count = privileged_info["red_stems_in_box_count"]
        green_stems_in_box_count = privileged_info["green_stems_in_box_count"]
         
        
        ## Rewards
        r_red = - np.tanh(20 * min_red_dist) if min_red_dist != float('inf') else 0.0
        r_alignment = - np.tanh(60 * privileged_info["radial_dist"]) if min_red_dist != float('inf') else 0.0
        r_in_box = 0.0 if red_stems_in_box_count == 1 and green_stems_in_box_count == 0 else -1.0

        
        ## Penalties
        r_green_in_box_penalty = -1.0 if green_stems_in_box_count > 0 else 0.0
        r_col = -1.0 if collision_detected else 0.0
        r_dist = - np.tanh(5 * privileged_info["total_displacement"])

        # Penalize large actions and large changes in actions (reduce shakiness)
        # r_energy = -np.tanh(0.5*np.linalg.norm(action[:-1]))  # Exclude the grasp action
        # r_smooth = -np.tanh(0.5*np.linalg.norm(action[:-1] - self.prev_action[:-1]))
        r_energy = -np.linalg.norm(action[:-1])  # Exclude the grasp action
        r_smooth = -np.linalg.norm(action[:-1] - self.prev_action[:-1])
        self.prev_action = action

        if np.array_equal(self.gripper_vec, self.gripper_dict["closing"]) or np.array_equal(self.gripper_vec, self.gripper_dict["opening"]):
            r_gripper = -1.0
        else:
            r_gripper = 0.0

        # Reward for attempting to close gripper when very close to a red strawberry
        r_attempt_close = 0.0
        GRASP_ATTEMPT_DISTANCE_THRESHOLD = 0.03 # meters (3cm)
        if np.array_equal(self.gripper_vec, self.gripper_dict["closing"]): # Gripper is currently executing a close action
            if min_red_dist < GRASP_ATTEMPT_DISTANCE_THRESHOLD:
                r_attempt_close = 1.0 # Positive reinforcement for trying to close at the right spot

        # Check for successful grasp and handle strawberry removal
        r_grasp = 0.0
        r_bad_grasp = -float(bad_grasp)
        grasped_idx = privileged_info.get("grasped_idx", None)
        
        if good_grasp and (not bad_grasp) and grasped_idx is not None:
            curr_pos = self.data.sensor(f"block{grasped_idx}_pos").data
            init_pos = self.red_positions[grasped_idx]
            dist_from_init = np.linalg.norm(curr_pos - init_pos)

            if dist_from_init < 0.05:
                r_grasp = 1.0
                # Only pay r_grasp once per strawberry
                if grasped_idx not in self._grasped_pending:
                    self._blocks_picked += 1
                    self._grasped_pending.add(grasped_idx)
                    # Schedule disappearance after N steps; keep visuals/collisions/rewards unchanged until then
                    self._pending_removals[grasped_idx] = int(getattr(self, "disappear_delay_steps", 8))
            else:
                r_grasp = 0.0

        # Penalty for being alive
        r_alive = -1.0 
        
        if len(self.red_blocks) == 0:
            completed = True
        else:
            completed = False
        info = {}
        
        rewards = {'r_grasp': r_grasp, 
                'r_red': r_red, 
                'r_alignment': r_alignment,
                'r_in_box': r_in_box,
                'r_green_in_box_penalty': r_green_in_box_penalty,
                'r_col': r_col, 
                'r_dist': r_dist, 
                'r_attempt_close': r_attempt_close, 
                'r_bad_grasp': r_bad_grasp, 
                'r_energy': r_energy, 
                'r_smooth': r_smooth,
                'r_gripper': r_gripper,
                'r_alive': r_alive}
        reward_scales = {'r_grasp': 8.0, 
                        'r_red': 4.0, 
                        'r_alignment': 1.0,
                        'r_in_box': 1.0,
                        'r_green_in_box_penalty': 1.0,
                        'r_col': 1.0, 
                        'r_dist': 1.0, 
                        'r_attempt_close': 2.0, 
                        'r_bad_grasp': 2.0, 
                        'r_energy': 1.0, 
                        'r_smooth': 1.0,
                        'r_gripper': 0.2,
                        'r_alive': 0.0}
        rewards = {k: v * reward_scales[k] for k, v in rewards.items()}
        reward = np.clip(sum(rewards.values()), -1e4, 1e4)
        info = rewards
        info['blocks_picked'] = self._blocks_picked

        info['success'] = completed
        return reward, info
    
    def _tick_removal_timers(self):
        if not self._pending_removals:
            return
        # Decrement counters
        for k in list(self._pending_removals.keys()):
            self._pending_removals[k] -= 1
        # Apply any that reached zero
        due = [k for k, t in self._pending_removals.items() if t <= 0]
        for idx in due:
            self._apply_removal(idx)
            self._pending_removals.pop(idx, None)
            self._grasped_pending.discard(idx)

    def _apply_removal(self, idx: int):
        # Hide all geoms for this strawberry (block{idx}, block{idx}_big, block{idx}_small)
        for suffix in ["", "_big", "_small"]:
            body_name = f"block{idx}{suffix}"
            try:
                body = self.model.body(body_name)
            except Exception:
                continue
            geom_start = self.model.body_geomadr[body.id]
            geom_count = self.model.body_geomnum[body.id]
            for k in range(geom_count):
                geom_id = geom_start + k
                self.model.geom_group[geom_id] = 3
                self.model.geom_contype[geom_id] = 0
                self.model.geom_conaffinity[geom_id] = 0

        # Remove from active lists (matches your current behavior)
        if hasattr(self, "active_indices") and (idx in self.active_indices):
            self.active_indices = np.delete(self.active_indices, np.where(self.active_indices == idx))
        if hasattr(self, "red_blocks") and (idx in self.red_blocks):
            self.red_blocks.remove(idx)
