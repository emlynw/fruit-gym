import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import cv2
from fruit_gym import envs
import numpy as np
from mujoco_ar import MujocoARConnector
import time
from scipy.spatial.transform import Rotation

def main():
    render_mode = "rgb_array"
    ee_dof = 6
    cameras = ['wrist1', 'wrist2']
    env = gym.make("PickMultiStrawbEnv", pos_scale=0.2, render_mode=render_mode, randomize_domain=True, ee_dof=ee_dof)
    env = TimeLimit(env, max_episode_steps=1000)    
    waitkey = 10
    connector = MujocoARConnector()
    resize_resolution = (640, 640)

    # Start the connector
    connector.start()
    data = connector.get_latest_data()  # Returns {"position": (3, 1), "rotation": (3, 3), "button": bool, "toggle": bool}
    while data['position'] is None:
        data = connector.get_latest_data()
        time.sleep(1)
        print("Waiting for AR data...")

    while True:
        # reset the environment
        i=0
        terminated = False
        truncated = False
        obs, info = env.reset()
        connector.reset_position()
        action = np.array([0.0]*(ee_dof+1))

        while not terminated and not truncated:
            pos = data["position"]
            pos_new = np.array([pos[2], -pos[1], -pos[0]])
            print(f"Position: {pos}, Button: {data['button']}")
            rot = []
            grasp = [float(data["button"])]
            if ee_dof == 4:
                r = Rotation.from_matrix(data["rotation"])
                angles = r.as_euler("xyz", degrees=False)
                rot = [-angles[2]]
            elif ee_dof == 6:
                r = Rotation.from_matrix(data["rotation"])
                angles = r.as_euler("xyz", degrees=False)
                rot = angles
                rot[2] = -rot[2]

            action = np.concatenate((pos_new, rot, grasp))
            obs, reward, terminated, truncated, info = env.step(action)
            cv2.imshow("wrist2", cv2.resize(cv2.cvtColor(obs['images']['wrist2'], cv2.COLOR_RGB2BGR), resize_resolution))
            cv2.imshow("wrist1", cv2.resize(cv2.cvtColor(obs["images"]["wrist1"], cv2.COLOR_RGB2BGR), resize_resolution))
            cv2.waitKey(waitkey)

            i+=1
        
if __name__ == "__main__":
    main()
