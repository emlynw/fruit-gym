#!/usr/bin/env python3
"""
Quick MuJoCo-pixel FPS benchmark

Runs N episodes with random actions and prints the achieved frames-per-second.
Everything except the GUI / video recorder is left unchanged so you can
compare numbers directly against your training loop.
"""
import time
import os
os.environ['MUJOCO_GL'] = 'egl'
import argparse
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from fruit_gym import envs                 # noqa: F401  (registers envs)
from wrappers import RotateImage, SERLObsWrapper
import numpy as np



def make_env(camera_res: int = 480,
             max_steps: int = 500) -> gym.Env:
    """Instantiate the PickMultiStrawbEnv with the same wrappers you use."""
    env = gym.make(
        "PickMultiStrawbPhysicsEnv",
        randomize_domain=True,
        image_obs=False,
        reward_type="dense",
        ee_dof=6,
        width=camera_res,
        height=camera_res,
        gripper_pause=False,
    )
    env = TimeLimit(env, max_episode_steps=max_steps)
    return env


def benchmark_fps(episodes: int = 5,
                  max_steps: int = 500,
                  camera_res: int = 480) -> float:
    env = make_env(camera_res, max_steps)
    total_frames = 0
    t0 = time.perf_counter()

    for ep in range(episodes):
        obs, info = env.reset(seed=np.random.randint(0, 2**32 - 1))
        terminated = truncated = False
        while not (terminated or truncated):
            action = env.action_space.sample()          # random action
            obs, reward, terminated, truncated, info = env.step(action)
            total_frames += 1

    elapsed = time.perf_counter() - t0
    env.close()
    fps = total_frames / elapsed
    print(
        f"Collected {total_frames} frames in {elapsed:.2f}s "
        f"→ {fps:.1f} FPS "
        f"({episodes} episodes, ≤{max_steps} steps each, {camera_res}px)"
    )
    return fps


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pixel-env FPS benchmark")
    parser.add_argument("--episodes", type=int, default=5,
                        help="number of rollouts to run (default: 5)")
    parser.add_argument("--max-steps", type=int, default=100,
                        help="max steps per episode (default: 500)")
    parser.add_argument("--res", type=int, default=112,
                        help="camera resolution (square) (default: 480)")
    args = parser.parse_args()

    benchmark_fps(args.episodes, args.max_steps, args.res)
