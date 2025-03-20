import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import cv2
from fruit_gym import envs
import numpy as np
import time

# Global variables to capture mouse movement
mouse_x, mouse_y = 0, 0  # Track mouse position

def mouse_callback(event, x, y, flags, param):
    global mouse_x, mouse_y
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x, mouse_y = x, y

def main():
    render_mode = "rgb_array"
    env = gym.make("PickStrawbEnv", render_mode=render_mode, randomize_domain=True, reward_type="sparse", ee_dof=4)
    env = TimeLimit(env, max_episode_steps=500)    
    waitkey = 50
    resize_resolution = (480, 480)

    # Define the range for absolute movement control
    max_speed = 1.0  # Maximum speed in any direction
    rot_speed = 1.0  # Maximum rotation speed

    # Set up mouse callback
    cv2.namedWindow("wrist2")
    cv2.setMouseCallback("wrist2", mouse_callback)
    
    while True:
        terminated = False
        truncated = False
        obs, info = env.reset()
        i=0
        
        while not (terminated or truncated):
            step_start_time = time.time()
            i+=1
            # Display the environment
            if render_mode == "rgb_array":
                cv2.imshow("wrist2", cv2.resize(cv2.cvtColor(obs['images']['wrist2'], cv2.COLOR_RGB2BGR), resize_resolution))
                # Rotate wrist1 by 180 degrees
                wrist1 = cv2.rotate(obs['images']['wrist1'], cv2.ROTATE_180)
                cv2.imshow("wrist1", cv2.resize(cv2.cvtColor(wrist1, cv2.COLOR_RGB2BGR), resize_resolution))
                cv2.imshow("front", cv2.resize(cv2.cvtColor(obs["images"]["front"], cv2.COLOR_RGB2BGR), resize_resolution))
            
            # Calculate movement based on absolute mouse position within window
            move_left_right = ((mouse_x / resize_resolution[0]) * 2 - 1) * max_speed
            move_up_down = -((mouse_y / resize_resolution[1]) * 2 - 1) * max_speed

            # Define movement actions for W and S keys (forward/backward)
            key = cv2.waitKey(waitkey) & 0xFF
            move_action = np.array([0, move_left_right, move_up_down, 0.0, 0.0])  # Default move

            if key == ord('w'):
                move_action[0] = max_speed  # Forward
            elif key == ord('s'):
                move_action[0] = -max_speed   # Backward
            elif key == ord('a'):
                move_action[3] = rot_speed
            elif key == ord('d'):
                move_action[3] = -rot_speed

            # Toggle gripper state with spacebar
            if key == ord(' '):
                move_action[-1] = 1.0
            elif key == ord('c'):
                move_action[-1] = -1.0

            # Perform the action in the environment
            step_time = time.time()-step_start_time
            if step_time < waitkey/1000:
                time.sleep(waitkey/1000 - step_time)
            obs, reward, terminated, truncated, info = env.step(move_action)

            # Reset environment on 'R' key press
            if key == ord('r'):
                print("Resetting environment...")
                i=0
                obs, info = env.reset()  # Reset the environment
                continue  # Start the loop again after reset

            # Exit on 'ESC' key
            if key == 27:  # ESC key
                print("Exiting...")
                env.close()
                cv2.destroyAllWindows()
                return

if __name__ == "__main__":
    main()
