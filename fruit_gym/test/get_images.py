import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import cv2
from fruit_gym import envs
import numpy as np
from gamepad_wrapper import GamepadIntervention
from wrappers import VideoRecorder, RotateImage, SERLObsWrapper
import time
import os

# --- add near the top of gamepad_teleop.py ---
def set_global_seed(seed: int) -> None:
    import os, random, numpy as np
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

def main():

    camera_res = 480
    image_num = 8
    cameras = ['wrist1', 'wrist2', 'front']
    proprio_keys = ["tcp_pose", "gripper_pos"]
    dir = os.path.dirname(__file__)


    env = gym.make("PickMultiStrawbHardEnv", physics_dt=0.001, randomize_domain=True, reward_type="dense", cameras=cameras,ee_dof=6, width=camera_res, 
                   height=camera_res, gripper_pause=False, use_potential_rewards=True, include_privileged_obs=True)
    env = SERLObsWrapper(env, proprio_keys=proprio_keys)
    env = RotateImage(env, pixel_key="wrist1")


    reset_time = time.time()
    set_global_seed(0)
    obs, info = env.reset(seed=0)

    images_dir = os.path.join(dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    for camera in cameras:
        image = obs[camera]
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        filename = os.path.join(images_dir, f"{image_num}_{camera}.png")
        cv2.imwrite(filename, image_bgr)
    print(f"reset took {time.time() - reset_time:.4f} seconds")


if __name__ == "__main__":
    main()
