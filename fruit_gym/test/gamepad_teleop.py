import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import cv2
from fruit_gym import envs
import numpy as np
from gamepad_wrapper import GamepadIntervention
from wrappers import VideoRecorder, RotateImage, SERLObsWrapper
import time
import os

def main():
    record = True
    camera_res = 480
    video_res = 480
    cameras = ['wrist1', 'wrist2']
    proprio_keys = ["tcp_pose", "gripper_pos"]
    display_res = (640, 640)
    fps = 20  # Frames per second for video recording
    num_episodes = 10
    dir = os.path.dirname(__file__)
    video_dir = os.path.join(dir, 'videos')
    waitkey = 10

    env = gym.make("PickStrawbEnv", randomize_domain=True, reward_type="dense", ee_dof=6, width=camera_res, height=camera_res, gripper_pause=False)
    env = TimeLimit(env, max_episode_steps=500)
    env = SERLObsWrapper(env, proprio_keys=proprio_keys)
    env = RotateImage(env, pixel_key="wrist1")
    env = GamepadIntervention(env)
    if record:
        for image_name in cameras:
                crop_res = env.observation_space[image_name].shape[0]
                env = VideoRecorder(env, video_dir, camera_name=image_name, crop_resolution=crop_res, resize_resolution=video_res, fps=fps, record_every=1)


    for episode in range(num_episodes):
        terminated = False
        truncated = False
        obs, info = env.reset()

        # Display and record the initial frame for each camera.
        print(f"Press any key to start episode {episode}")
        for camera in cameras:
            frame = cv2.resize(cv2.cvtColor(obs[camera], cv2.COLOR_RGB2BGR), display_res)
            cv2.imshow(camera, frame)
        cv2.waitKey(0)  # Wait for a key press to begin the episode

        while not terminated and not truncated:
            step_start_time = time.time()
            for camera in cameras:
                frame = cv2.resize(cv2.cvtColor(obs[camera], cv2.COLOR_RGB2BGR), display_res)
                cv2.imshow(camera, frame)
                cv2.waitKey(waitkey)

            action = np.zeros_like(env.action_space.sample())
            if "intervene_action" in info:
                action = info['intervene_action']

            obs, reward, terminated, truncated, info = env.step(action)
            step_time = time.time() - step_start_time
            if step_time < 0.05:
                time.sleep(0.05 - step_time)



if __name__ == "__main__":
    main()
