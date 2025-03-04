import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import cv2
from fruit_gym import envs
import numpy as np
from gamepad_wrapper import GamepadIntervention
import time

def main():
    render_mode = "rgb_array"
    env = gym.make("PickStrawbEnv", render_mode=render_mode, randomize_domain=True, reward_type="dense", ee_dof=6)
    env = GamepadIntervention(env)
    env = TimeLimit(env, max_episode_steps=500)
    waitkey = 10
    cameras = ['wrist1', 'wrist2', 'front']
    resize_resolution = (480, 480)
    fps = 20.0  # Frames per second for video recording

    # Create one continuous video writer for each camera.
    writers = {}
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    for camera in cameras:
        filename = f"{camera}.mp4"
        writers[camera] = cv2.VideoWriter(filename, fourcc, fps, resize_resolution)

    num_episodes = 10
    for episode in range(num_episodes):
        print(f"Starting episode {episode}")
        terminated = False
        truncated = False
        obs, info = env.reset()

        # Display and record the initial frame for each camera.
        for camera in cameras:
            if camera == "wrist1":
                frame = cv2.resize(cv2.cvtColor(cv2.rotate(obs['images'][camera], cv2.ROTATE_180), cv2.COLOR_RGB2BGR), resize_resolution)
            else:
                frame = cv2.resize(cv2.cvtColor(obs['images'][camera], cv2.COLOR_RGB2BGR), resize_resolution)
            cv2.imshow(camera, frame)
            writers[camera].write(frame)
        cv2.waitKey(0)  # Wait for a key press to begin the episode

        while not terminated and not truncated:
            step_start_time = time.time()
            for camera in cameras:
                if camera == "wrist1":
                    frame = cv2.resize(cv2.cvtColor(cv2.rotate(obs['images'][camera], cv2.ROTATE_180), cv2.COLOR_RGB2BGR), resize_resolution)
                else:
                    frame = cv2.resize(cv2.cvtColor(obs['images'][camera], cv2.COLOR_RGB2BGR), resize_resolution)
                cv2.imshow(camera, frame)
                writers[camera].write(frame)
                cv2.waitKey(waitkey)

            action = np.zeros_like(env.action_space.sample())
            if "intervene_action" in info:
                action = info['intervene_action']

            obs, reward, terminated, truncated, info = env.step(action)
            print(reward)
            step_time = time.time() - step_start_time
            if step_time < 0.05:
                time.sleep(0.05 - step_time)

    # Release all video writers after completing the episodes.
    for writer in writers.values():
        writer.release()

if __name__ == "__main__":
    main()
