import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import cv2
# Assuming fruit_gym and wrappers are in the python path
from fruit_gym import envs
import numpy as np
import time
import os
from wrappers import VideoRecorder, RotateImage, SERLObsWrapper
from functools import partial

# Global variables to capture mouse movement and display index
mouse_x, mouse_y = 0, 0
display_start_idx = 0  # Index of the first environment to display

def mouse_callback(event, x, y, flags, param):
    """CV2 mouse callback to update global mouse coordinates."""
    global mouse_x, mouse_y
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x, mouse_y = x, y

def main():
    # --- Configuration ---
    record = False
    num_envs = 12  # Number of parallel environments (Recommend >= 2)
    seed_list = np.random.randint(0, 2**32 - 1, size=num_envs).tolist()
    camera_res = 480
    video_res = 480
    cameras = ['wrist1', 'wrist2']
    proprio_keys = None
    display_res = (480, 480)
    fps = 20
    dir = os.path.dirname(__file__)
    video_dir = os.path.join(dir, 'videos')
    waitkey = 10

    if num_envs < 2:
        print("Warning: num_envs is less than 2. Side-by-side display will show the same environment twice.")

    # --- Wrapper Preparation ---
    wrapper_list = [
        partial(TimeLimit, max_episode_steps=500),
        partial(SERLObsWrapper, proprio_keys=proprio_keys),
        partial(RotateImage, pixel_key="wrist1"),
    ]
    if record:
        for image_name in cameras:
            crop_res = camera_res
            wrapper_list.append(
                partial(VideoRecorder, video_dir=video_dir, camera_name=image_name,
                        crop_resolution=crop_res, resize_resolution=video_res,
                        fps=fps, record_every=1)
            )

    # --- Vectorized Environment Creation ---
    envs = gym.make_vec(
        "PickMultiStrawbPhysicsEnv",
        num_envs=num_envs,
        vectorization_mode="async",
        vector_kwargs={},
        randomize_domain=True,
        cameras=cameras,
        reward_type="dense",
        ee_dof=4,
        width=camera_res,
        height=camera_res,
        gripper_pause=False,
        wrappers=wrapper_list,
    )

    max_speed = 1.0
    rot_speed = 1.0

    # Set up mouse callback for all display windows
    for camera in cameras:
        cv2.namedWindow(camera)
        cv2.setMouseCallback(camera, mouse_callback)

    global display_start_idx
    print(f"--- Controls ---")
    print(f"Displaying two environments side-by-side.")
    print(f"Mouse: Control X/Y position (relative to double-wide window)")
    print(f"W/S: Move forward/backward")
    print(f"A/D: Rotate")
    print(f"Space/C: Control gripper")
    print(f"R: Reset ALL environments")
    print(f"',' (comma): View previous pair of environments")
    print(f"'.' (period): View next pair of environments")
    print(f"ESC: Exit")
    print(f"----------------")

    obs, infos = envs.reset()

    while True:
        step_start_time = time.time()

        # Calculate single action based on user input relative to the double-wide display
        total_display_width = display_res[0] * 2
        move_left_right = ((mouse_x / total_display_width) * 2 - 1) * max_speed
        move_up_down = -((mouse_y / display_res[1]) * 2 - 1) * max_speed

        single_action = np.zeros(envs.single_action_space.shape, dtype=envs.single_action_space.dtype)
        single_action[1] = move_left_right
        single_action[2] = move_up_down

        key = cv2.waitKey(waitkey) & 0xFF

        if key == ord('w'): single_action[0] = max_speed
        elif key == ord('s'): single_action[0] = -max_speed
        elif key == ord('a'): single_action[3] = rot_speed
        elif key == ord('d'): single_action[3] = -rot_speed
        if key == ord(' '): single_action[-1] = 1.0
        elif key == ord('c'): single_action[-1] = -1.0

        if key == ord(','):
            display_start_idx = (display_start_idx - 1 + num_envs) % num_envs
            print(f"Displaying Environments: {display_start_idx} and {(display_start_idx + 1) % num_envs}")
        elif key == ord('.'):
            display_start_idx = (display_start_idx + 1) % num_envs
            print(f"Displaying Environments: {display_start_idx} and {(display_start_idx + 1) % num_envs}")

        if key == ord('r'):
            print("Resetting all environments...")
            obs, infos = envs.reset()
            continue
        if key == 27:
            print("Exiting...")
            break

        actions = np.stack([single_action] * num_envs)
        obs, rewards, terminations, truncations, infos = envs.step(actions)

        if np.any(terminations) or np.any(truncations):
            finished_envs = np.where(terminations | truncations)[0]
            # This can be noisy, uncomment if you want to see every auto-reset
            # print(f"Environments {finished_envs} finished and were auto-reset.")

        # --- Side-by-Side Display Logic ---
        idx1 = display_start_idx
        idx2 = (display_start_idx + 1) % num_envs

        for camera in cameras:
            # Get frames for both environments
            frame1_raw = obs[camera][idx1]
            frame2_raw = obs[camera][idx2]

            # Process and add text to the first frame
            frame1 = cv2.resize(cv2.cvtColor(frame1_raw, cv2.COLOR_RGB2BGR), display_res)
            reward1 = rewards[idx1]
            cv2.putText(frame1, f"Env: {idx1} | R: {reward1:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Process and add text to the second frame
            frame2 = cv2.resize(cv2.cvtColor(frame2_raw, cv2.COLOR_RGB2BGR), display_res)
            reward2 = rewards[idx2]
            cv2.putText(frame2, f"Env: {idx2} | R: {reward2:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Stack the frames horizontally
            composite_frame = np.hstack((frame1, frame2))
            
            # Show the combined image
            cv2.imshow(camera, composite_frame)

        step_time = time.time() - step_start_time
        if step_time < waitkey / 1000:
            time.sleep(waitkey / 1000 - step_time)

    envs.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
