import numpy as np
import random
from typing import Dict, Any

def lighting_noise(env: Any) -> None:
    """
    Randomize lighting conditions using domain randomization settings.
    
    This function:
      1. Adds random offsets to the light position.
      2. Randomizes diffuse, ambient, and specular properties.
      3. Randomizes headlight parameters.
    
    Args:
        env: The environment instance (must have attributes: cfg, model, init_light_pos, 
             init_headlight_diffuse, init_headlight_ambient, init_headlight_specular).
    """
    dr_cfg: Dict[str, Any] = env.cfg.get("domain_randomization", {})
    lighting_cfg: Dict[str, Any] = dr_cfg.get("lighting", {})
    if not lighting_cfg.get("enabled", False):
        return

    # Randomize light position.
    pos_low = lighting_cfg.get("position_range_low", [-0.8, -0.5, -0.05])
    pos_high = lighting_cfg.get("position_range_high", [1.2, 0.5, 0.2])
    light_pos_noise = np.random.uniform(low=pos_low, high=pos_high, size=3)
    env.model.body_pos[env.model.body('light0').id] = env.init_light_pos + light_pos_noise

    # Vary brightness: 50% chance darker or brighter.
    if random.random() < 0.5:
        diffuse_low = lighting_cfg.get("diffuse_range", {}).get("low", [0.05, 0.05, 0.05])
        diffuse_high = lighting_cfg.get("diffuse_range", {}).get("high", [0.3, 0.3, 0.3])
        ambient_low = lighting_cfg.get("ambient_range", {}).get("low", [0.0, 0.0, 0.0])
        ambient_high = lighting_cfg.get("ambient_range", {}).get("high", [0.2, 0.2, 0.2])
        light_diffuse_noise = np.random.uniform(low=diffuse_low, high=diffuse_high, size=3)
        light_ambient_noise = np.random.uniform(low=ambient_low, high=ambient_high, size=3)
    else:
        light_diffuse_noise = np.random.uniform(low=[0.1, 0.1, 0.1], high=[1.0, 1.0, 1.0], size=3)
        light_ambient_noise = np.random.uniform(low=[0.0, 0.0, 0.0], high=[0.5, 0.5, 0.5], size=3)

    env.model.light_diffuse[0] = light_diffuse_noise
    env.model.light_ambient[0] = light_ambient_noise

    light_specular_noise = np.random.uniform(
        low=lighting_cfg.get("specular_range", {}).get("low", [0.0, 0.0, 0.0]),
        high=lighting_cfg.get("specular_range", {}).get("high", [0.5, 0.5, 0.5]),
        size=3
    )
    env.model.light_specular[0] = light_specular_noise

    # Headlight noise.
    if random.random() < 0.5:
        light_factor = -1.0
        head_diffuse = np.random.uniform(low=0.0, high=0.1, size=3)
        head_ambient = np.random.uniform(low=0.0, high=0.1, size=3)
        head_specular = np.random.uniform(low=0.0, high=0.1, size=3)
    else:
        light_factor = 1.0
        head_diffuse = np.random.uniform(low=0.0, high=0.1, size=3)
        head_ambient = np.random.uniform(low=0.0, high=0.1, size=3)
        head_specular = np.random.uniform(low=0.0, high=0.1, size=3)

    env.model.vis.headlight.diffuse = env.init_headlight_diffuse + light_factor * head_diffuse
    env.model.vis.headlight.ambient = env.init_headlight_ambient + light_factor * head_ambient
    env.model.vis.headlight.specular = env.init_headlight_specular + light_factor * head_specular


def action_scale_noise(env: Any) -> None:
    """
    Randomize action scaling factors (pos_scale and rot_scale) based on configuration.
    
    Args:
        env: The environment instance (must have attributes: cfg, pos_scale, rot_scale).
    """
    dr_cfg: Dict[str, Any] = env.cfg.get("domain_randomization", {})
    action_cfg: Dict[str, Any] = dr_cfg.get("action_scale", {})
    if not action_cfg.get("enabled", False):
        return

    pos_scale_range = action_cfg.get("pos_scale_range", [0.0, 0.0])
    rot_scale_range = action_cfg.get("rot_scale_range", [0.0, 0.0])
    env.pos_scale = random.uniform(*pos_scale_range)
    env.rot_scale = random.uniform(*rot_scale_range)


def initial_state_noise(env: Any) -> None:
    """
    Apply noise to the initial end-effector state.
    
    Args:
        env: The environment instance (must have attributes: cfg, data, _PANDA_XYZ).
    """
    dr_cfg: Dict[str, Any] = env.cfg.get("domain_randomization", {})
    state_cfg: Dict[str, Any] = dr_cfg.get("initial_state", {})
    if not state_cfg.get("enabled", False):
        return

    ee_noise_low = state_cfg.get("ee_noise_low", [0.0, 0.0, 0.0])
    ee_noise_high = state_cfg.get("ee_noise_high", [0.0, 0.0, 0.0])
    ee_noise = np.random.uniform(low=ee_noise_low, high=ee_noise_high, size=3)
    env.data.mocap_pos[0] = env._PANDA_XYZ + ee_noise


def camera_noise(env: Any) -> None:
    """
    Apply noise to camera positions and orientations based on configuration.
    
    Args:
        env: The environment instance (must have attributes: cfg, cameras, model, etc.).
    """
    dr_cfg: Dict[str, Any] = env.cfg.get("domain_randomization", {})
    cam_cfg: Dict[str, Any] = dr_cfg.get("cameras", {})
    if not cam_cfg.get("enabled", False):
        return

    for cam_name in env.cameras:
        cam_settings: Dict[str, Any] = cam_cfg.get(cam_name, {})
        pos_noise_low = cam_settings.get("position_noise_low", [0.0, 0.0, 0.0])
        pos_noise_high = cam_settings.get("position_noise_high", [0.0, 0.0, 0.0])
        quat_noise_range = cam_settings.get("quat_noise_range", [0.0, 0.0])

        cam_pos_noise = np.random.uniform(low=pos_noise_low, high=pos_noise_high, size=3)
        env.model.body_pos[env.model.body(cam_name).id] = getattr(env, f"{cam_name}_pos") + cam_pos_noise

        cam_quat_noise = np.random.uniform(low=quat_noise_range[0], high=quat_noise_range[1], size=4)
        new_cam_quat = getattr(env, f"{cam_name}_quat") + cam_quat_noise
        new_cam_quat /= np.linalg.norm(new_cam_quat)
        env.model.body_quat[env.model.body(cam_name).id] = new_cam_quat


def floor_noise(env: Any) -> None:
    """
    Randomize the floor texture.
    
    Args:
        env: The environment instance (must have attribute: floor_tex_ids, model).
    """
    floor_tex_id = np.random.choice(env.floor_tex_ids)
    env.model.mat_texid[env.model.mat('floor').id] = floor_tex_id


def skybox_noise(env: Any) -> None:
    """
    Randomize the skybox texture.
    
    Args:
        env: The environment instance (must have attribute: skybox_tex_ids, model, _viewer).
    """
    skybox_tex_id = np.random.choice(env.skybox_tex_ids)
    start_idx = env.model.name_texadr[skybox_tex_id]
    end_idx = env.model.name_texadr[skybox_tex_id + 1] - 1 if skybox_tex_id + 1 < len(env.model.name_texadr) else None
    texture_name = env.model.names[start_idx:end_idx].decode('utf-8')
    if 'sky' not in texture_name:
        env.model.geom('floor').group = 3
    else:
        env.model.geom('floor').group = 0
    env._viewer.model.tex_adr[0] = env.model.tex_adr[skybox_tex_id]
