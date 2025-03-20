import gymnasium as gym
import numpy as np
import cv2
import imageio
import os
from gymnasium.spaces import flatten_space, flatten

class VideoRecorder(gym.Wrapper):
    """Wrapper for rendering and saving rollouts to disk from a specific camera."""

    def __init__(
        self,
        env,
        save_dir,
        crop_resolution,
        resize_resolution,
        camera_name="wrist2",
        fps=10,
        current_episode=0,
        record_every=2,
    ):
        super().__init__(env)

        self.save_dir = save_dir
        self.camera_name = camera_name
        os.makedirs(save_dir, exist_ok=True)
        num_vids = len([f for f in os.listdir(save_dir) if f.endswith(f"{camera_name}.mp4")])
        print(f"num_vids: {num_vids}")
        current_episode = num_vids * record_every

        if isinstance(resize_resolution, int):
            self.resize_resolution = (resize_resolution, resize_resolution)
        if isinstance(crop_resolution, int):
            self.crop_resolution = (crop_resolution, crop_resolution)

        self.resize_h, self.resize_w = self.resize_resolution
        self.crop_h, self.crop_w = self.crop_resolution
        self.fps = fps
        self.enabled = True
        self.current_episode = current_episode
        self.record_every = record_every
        self.frames = []

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        if self.current_episode % self.record_every == 0:
            frame = observation[self.camera_name].copy()
            
            if self.crop_resolution is not None:
                if frame.shape[:2] != (self.crop_h, self.crop_w):
                    center = frame.shape
                    x = center[1] // 2 - self.crop_w // 2
                    y = center[0] // 2 - self.crop_h // 2
                    frame = frame[int(y):int(y + self.crop_h), int(x):int(x + self.crop_w)]

            if self.resize_resolution is not None:
                if frame.shape[:2] != (self.resize_h, self.resize_w):
                    frame = cv2.resize(
                        frame,
                        dsize=(self.resize_w, self.resize_h),
                        interpolation=cv2.INTER_CUBIC,
                    )
            frame  = np.ascontiguousarray(frame)
            # cv2.putText(
            #     frame,
            #     f"{reward:.3f}",
            #     (10, 40),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     0.5,
            #     (0, 255, 0),
            #     1,
            #     cv2.LINE_AA,
            # )

            self.frames.append(frame)

        if terminated or truncated:
            if self.current_episode % self.record_every == 0:
                if info['success']:
                    filename = os.path.join(self.save_dir, f"{self.current_episode}_success_{self.camera_name}.mp4")
                else:
                    filename = os.path.join(self.save_dir, f"{self.current_episode}_failure_{self.camera_name}.mp4")
                imageio.mimsave(filename, self.frames, fps=self.fps)
                self.frames = []

            self.current_episode += 1

        return observation, reward, terminated, truncated, info
    
class RotateImage(gym.ObservationWrapper):
    """Rotate the pixel observation by 180 degrees."""

    def __init__(self, env, pixel_key='pixels'):
        super().__init__(env)
        self.pixel_key = pixel_key

        # Optionally, update the observation space if needed.
        # Since a 180° rotation doesn't change the image shape,
        # we can just copy the existing space.
        self.observation_space = env.observation_space

    def observation(self, observation):
        # Extract the image from the observation using the specified key.
        image = observation[self.pixel_key]
        
         # Check if the image has a leading batch dimension.
        if image.shape[0] == 1:
            # Remove the batch dimension: shape becomes (height, width, 3)
            image = image[0]
            # Rotate the image by 180 degrees using OpenCV.
            rotated_image = cv2.rotate(image, cv2.ROTATE_180)
            # Re-add the batch dimension: shape becomes (1, height, width, 3)
            rotated_image = np.expand_dims(rotated_image, axis=0)
        else:
            # Otherwise, just rotate the image normally.
            rotated_image = cv2.rotate(image, cv2.ROTATE_180)
        
        
        # Replace the image in the observation with the rotated version.
        observation[self.pixel_key] = rotated_image
        return observation
    
class SERLObsWrapper(gym.ObservationWrapper):
    """
    This observation wrapper treat the observation space as a dictionary
    of a flattened state space and the images.
    """

    def __init__(self, env, proprio_keys=None):
        super().__init__(env)
        self.proprio_keys = proprio_keys
        if self.proprio_keys is None:
            self.proprio_keys = list(self.env.observation_space["state"].keys())

        self.proprio_space = gym.spaces.Dict(
            {key: self.env.observation_space["state"][key] for key in self.proprio_keys}
        )

        self.observation_space = gym.spaces.Dict(
            {
                "state": flatten_space(self.proprio_space),
                **(self.env.observation_space["images"]),
            }
        )

    def observation(self, obs):
        obs = {
            "state": flatten(
                self.proprio_space,
                {key: obs["state"][key] for key in self.proprio_keys},
            ),
            **(obs["images"]),
        }
        return obs

    def reset(self, **kwargs):
        obs, info =  self.env.reset(**kwargs)
        return self.observation(obs), info

def flatten_observations(obs, proprio_space, proprio_keys):
        obs = {
            "state": flatten(
                proprio_space,
                {key: obs["state"][key] for key in proprio_keys},
            ),
            **(obs["images"]),
        }
        return obs