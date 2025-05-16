import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import cv2
from fruit_gym import envs
import numpy as np
import time
import os
from wrappers import VideoRecorder, RotateImage, SERLObsWrapper

# Global variables to capture mouse movement
mouse_x, mouse_y = 0, 0  # Track mouse position

def mouse_callback(event, x, y, flags, param):
    global mouse_x, mouse_y
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x, mouse_y = x, y

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

    env = gym.make("PickMultiStrawbEnv", randomize_domain=True, reward_type="dense", ee_dof=4, width=camera_res, height=camera_res, gripper_pause=False)
    env = TimeLimit(env, max_episode_steps=500)
    env = SERLObsWrapper(env, proprio_keys=proprio_keys)
    env = RotateImage(env, pixel_key="wrist1")
    if record:
        for image_name in cameras:
                crop_res = env.observation_space[image_name].shape[0]
                env = VideoRecorder(env, video_dir, camera_name=image_name, crop_resolution=crop_res, resize_resolution=video_res, fps=fps, record_every=1)

    # Define the range for absolute movement control
    max_speed = 1.0  # Maximum speed in any direction
    rot_speed = 1.0  # Maximum rotation speed

    # Set up mouse callback
    cv2.namedWindow("wrist1")
    cv2.setMouseCallback("wrist1", mouse_callback)
    
    for episode in range(num_episodes):
        terminated = False
        truncated = False
        obs, info = env.reset()
        print(f"Press any key to start episode {episode}")
        for camera in cameras:
            frame = cv2.resize(cv2.cvtColor(obs[camera], cv2.COLOR_RGB2BGR), display_res)
            cv2.imshow(camera, frame)
        cv2.waitKey(0)  # Wait for a key press to begin the episode
        
        while not (terminated or truncated):
            step_start_time = time.time()
            # Display the environment
            for camera in cameras:
                frame = cv2.resize(cv2.cvtColor(obs[camera], cv2.COLOR_RGB2BGR), display_res)
                cv2.imshow(camera, frame)
            
            # Calculate movement based on absolute mouse position within window
            move_left_right = ((mouse_x / display_res[0]) * 2 - 1) * max_speed
            move_up_down = -((mouse_y / display_res[1]) * 2 - 1) * max_speed

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
            print(f"reward: {reward}")

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
