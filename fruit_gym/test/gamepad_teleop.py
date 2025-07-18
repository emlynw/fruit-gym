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
    record = False
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

    env = gym.make("PickMultiStrawbEnv", randomize_domain=True, discrete_gripper=False,reward_type="dense", ee_dof=6, width=camera_res, height=camera_res, gripper_pause=False)
    env = TimeLimit(env, max_episode_steps=250)
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
            action = np.zeros_like(env.action_space.sample())
            if "intervene_action" in info:
                action = info['intervene_action']

            obs, reward, terminated, truncated, info = env.step(action)
            for camera in cameras:
                frame = cv2.resize(cv2.cvtColor(obs[camera], cv2.COLOR_RGB2BGR), display_res)
                # Write reward on the frame
                cv2.putText(frame, f"Reward: {reward:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, f"Alignment: {info['r_alignment']:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, f"Distance: {info['r_red']:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, f"Gripper Box: {info['r_in_box']:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, f"Green Box Penalty: {info.get('r_green_in_box_penalty', 0):.2f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, f"Energy: {info['r_energy']:.2f}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, f"Smoothness: {info['r_smooth']:.2f}", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, f"collision: {info['r_col']:.2f}", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, f"displacement: {info['r_dist']:.2f}", (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)                   

                cv2.imshow(camera, frame)
                cv2.waitKey(waitkey)

            step_time = time.time() - step_start_time
            if step_time < 0.05:
                time.sleep(0.05 - step_time)



if __name__ == "__main__":
    main()
