import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import cv2
from fruit_gym import envs  # noqa: F401 - registers envs
import numpy as np
from gamepad_wrapper import GamepadIntervention
from wrappers import VideoRecorder, RotateImage, SERLObsWrapper
import time
import os

# NEW: depth adapter + wrapper
from depth_wrapper import VideoDepthAnythingAdapter, VideoDepthObsWrapper

def depth_to_grayscale(
    depth: np.ndarray,
    *,
    metric: bool = False,
    units: str = "m",        # "m" | "cm" | "mm"
    clip_m: float = 0.6,     # visualize range [0, clip_m] when metric=True
    out_size=None,           # (W, H) or None
    gamma: float = 1.0       # >1 boosts contrast near; <1 boosts far
) -> np.uint8:
    """
    Returns a uint8 grayscale image (H,W) for visualization.
    Near = bright, far = dark.

    - If metric=True, `depth` is interpreted in the given `units`, clipped to [0, clip_m].
    - If metric=False, `depth` is assumed already in [0,1] (clip applied).
    """
    d = np.asarray(depth, dtype=np.float32)
    # clean up NaNs/Infs
    d = np.where(np.isfinite(d), d, 0.0)

    if metric:
        # convert to meters
        if units == "mm":
            d_m = d / 1000.0
        elif units == "cm":
            d_m = d / 100.0
        else:
            d_m = d
        # normalize to [0,1] within [0, clip_m]
        d01 = np.clip(d_m / max(clip_m, 1e-6), 0.0, 1.0)
    else:
        # assume already [0,1]; clamp just in case
        d01 = np.clip(d, 0.0, 1.0)

    # Invert so near (small) => bright
    vis = 1.0 - d01

    # Optional gamma to shape contrast (gamma>1 brightens near more)
    if gamma != 1.0:
        vis = np.clip(vis, 0.0, 1.0) ** (1.0 / max(gamma, 1e-6))

    gray = (np.clip(vis, 0.0, 1.0) * 255.0).astype(np.uint8)

    if out_size is not None:
        gray = cv2.resize(gray, out_size, interpolation=cv2.INTER_LINEAR)
    return gray

def colorize_depth(
    depth: np.ndarray,
    out_size=None,
    *,
    metric: bool = False,       # True if `depth` is metric (cm or m)
    units: str = "cm",          # "cm" (default for metric VDA) or "m"
    clip_m: float = 0.2,        # visualize up to 1 meter
    invert: bool = True        # set True if you want near=bright and far=dark
) -> np.ndarray:
    """
    Colorize a single-channel depth map.

    Parameters
    ----------
    depth : (H, W) float array
        If metric=False: expected normalized [0,1].
        If metric=True: values in centimeters (default) or meters (units='m').
    out_size : (W, H) or None
        Optional output size for resizing.
    metric : bool
        Whether `depth` is metric. If True, clips to `clip_m` and maps to [0,1].
    units : str
        'cm' (assumed for metric VDA) or 'm'.
    clip_m : float
        Max visualized range in meters. Everything beyond is saturated.
    invert : bool
        If True, near->high value (bright), far->low (dark). Default False.

    Returns
    -------
    colored_bgr : (H, W, 3) uint8
        BGR image suitable for cv2.imshow
    """
    d = np.asarray(depth, dtype=np.float32)

    # Clean up NaNs/Infs
    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)

    if metric:
        # Convert to meters
        if units == "cm":
            d_m = d * 0.01
        elif units == "m":
            d_m = d
        else:
            raise ValueError("units must be 'cm' or 'm'")
        # Clip to visualization range and normalize to [0,1]
        d01 = np.clip(d_m, 0.0, clip_m) / max(clip_m, 1e-6)
    else:
        # Already normalized
        d01 = np.clip(d, 0.0, 1.0)

    if invert:
        d01 = 1.0 - d01

    # Map to 8-bit and colorize (TURBO)
    d8 = (d01 * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(d8, cv2.COLORMAP_TURBO)

    if out_size is not None:
        colored = cv2.resize(colored, out_size, interpolation=cv2.INTER_LINEAR)

    return colored


def main():
    record = False
    camera_res = 480
    video_res = 480
    metric_depth = False
    depth_units = "cm"
    cameras = ['wrist1', 'wrist2']
    proprio_keys = ["tcp_pose", "gripper_pos"]
    display_res = (640, 640)
    fps = 20
    num_episodes = 10
    dir = os.path.dirname(__file__)
    video_dir = os.path.join(dir, 'videos')
    waitkey = 10
    reward_type = "sparse"
    ee_dof = 6
    episode_reward = 0.0
    reward_scales = {
        # 'r_red': 100.0,
        # 'r_alignment': 100.0,
    }

    # --- Base env ---
    env = gym.make(
        "PickStrawbEnv",
        randomize_domain=True,
        reward_type=reward_type,
        ee_dof=ee_dof,
        width=camera_res,
        height=camera_res,
        gripper_pause=True
    )
    env = TimeLimit(env, max_episode_steps=2000)
    env = SERLObsWrapper(env, proprio_keys=proprio_keys)
    # rotate wrist1 before depth so depth sees the rotated image
    env = RotateImage(env, pixel_key="wrist1")
    env = GamepadIntervention(env)

    # --- Depth model (Video-Depth-Anything small, FP16) ---
    from depth_wrapper import VideoDepthAnythingAdapter

    adapter = VideoDepthAnythingAdapter(
        model_name="vits",
        input_size=224,
        use_metric=metric_depth,                       # switch to metric weights
        weights_path=None,                     # let it auto-pick
        weights_dir="/home/emlyn/video_depth_anything_models",
        fp32=False
    )

    env = VideoDepthObsWrapper(
        env,
        depth_estimator=adapter,
        rgb_keys=["wrist1", "wrist2"],
        normalize_depth=not adapter.use_metric  # don't normalize metric outputs
    )

    # Optional recording (still records RGB; you can extend VideoRecorder to also record *_depth if you want)
    if record:
        for image_name in cameras:
            crop_res = env.observation_space[image_name].shape[0]
            env = VideoRecorder(env, video_dir, camera_name=image_name,
                                crop_resolution=crop_res, resize_resolution=video_res,
                                fps=fps, record_every=1)

    for episode in range(num_episodes):
        terminated = False
        truncated = False
        obs, info = env.reset()

        print(f"Press any key to start episode {episode}")
        for camera in cameras:
            rgb = cv2.cvtColor(obs[camera], cv2.COLOR_RGB2BGR)
            rgb = cv2.resize(rgb, display_res, interpolation=cv2.INTER_LINEAR)
            depth_col = colorize_depth(obs[f"{camera}_depth"], metric=metric_depth, units=depth_units, out_size=display_res)
            stacked = np.hstack([rgb, depth_col])
            cv2.imshow(camera, stacked)
        key = cv2.waitKey(0)
        if key == ord('q'):
            return

        episode_reward = 0.0

        while not terminated and not truncated:
            step_start_time = time.time()
            action = np.zeros_like(env.action_space.sample())
            if "intervene_action" in info:
                action = info['intervene_action']

            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

            for camera in cameras:
                # RGB
                rgb = cv2.cvtColor(obs[camera], cv2.COLOR_RGB2BGR)
                rgb = cv2.resize(rgb, display_res, interpolation=cv2.INTER_LINEAR)

                # Depth (0..1 -> colormap)
                depth_col = colorize_depth(obs[f"{camera}_depth"], metric=metric_depth, units=depth_units, out_size=display_res)

                # Annotate RGB with reward
                cv2.putText(rgb, f"Reward: {reward:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                if reward_type == "dense":
                    cv2.putText(rgb, f"Alignment: {info['r_alignment']:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                    cv2.putText(rgb, f"Distance: {info['r_red']:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                    cv2.putText(rgb, f"Gripper Box: {info['r_in_box']:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                    cv2.putText(rgb, f"Green Box Penalty: {info.get('r_green_in_box_penalty', 0):.2f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                    cv2.putText(rgb, f"Energy: {info['r_energy']:.2f}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                    cv2.putText(rgb, f"Smoothness: {info['r_smooth']:.2f}", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                    cv2.putText(rgb, f"collision: {info['r_col']:.2f}", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                    cv2.putText(rgb, f"displacement: {info['r_dist']:.2f}", (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                    cv2.putText(rgb, f"Blocks Picked: {info.get('blocks_picked', 0)}", (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                    cv2.putText(rgb, f"Bad Grasp: {(info.get('r_bad_grasp', 0))}", (10, 330), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                    cv2.putText(rgb, f"Gripper: {info.get('r_gripper', 0):.2f}", (10, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                elif reward_type == "sparse":
                    cv2.putText(rgb, f"Dense Reward: {info['dense_reward']:.2f}", (10, 300),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

                # Stack RGB | DEPTH and show
                stacked = np.hstack([rgb, depth_col])
                cv2.imshow(camera, stacked)

                # non-blocking wait; 'q' to quit early
                k = cv2.waitKey(waitkey) & 0xFF
                if k == ord('q'):
                    return

            if terminated or truncated:
                for camera in cameras:
                    rgb = cv2.cvtColor(obs[camera], cv2.COLOR_RGB2BGR)
                    rgb = cv2.resize(rgb, display_res, interpolation=cv2.INTER_LINEAR)
                    cv2.putText(rgb, f"Score: {episode_reward:.3f}", (10, 180),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
                    depth_col = colorize_depth(obs[f"{camera}_depth"], metric=metric_depth, units=depth_units, out_size=display_res)
                    stacked = np.hstack([rgb, depth_col])
                    cv2.imshow(camera, stacked)
                key = cv2.waitKey(0)
                if key == ord('q'):
                    return

            # keep roughly 20 FPS loop pace (or whatever your control loop needs)
            step_time = time.time() - step_start_time
            target_dt = 1.0 / fps
            if step_time < target_dt:
                time.sleep(target_dt - step_time)

if __name__ == "__main__":
    main()
