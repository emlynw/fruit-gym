import gymnasium as gym
import numpy as np
import pygame

class GamepadExpert:
    """
    A gamepad interface that:
     - polls a joystick via pygame
     - maintains a 'gear' system for translation and rotation speeds
    """
    def __init__(self, 
                 translation_speeds=None,
                 rotation_speeds=None,
                 dead_zone=0.15):
        pygame.init()
        pygame.joystick.init()
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()

        # Default gears if none provided
        if translation_speeds is None:
            translation_speeds = [0.05, 0.1, 0.2, 0.5, 0.75, 1.0]
        if rotation_speeds is None:
            rotation_speeds = [0.05, 0.1, 0.2, 0.5, 0.75, 1.0]

        self.translation_speeds = translation_speeds
        self.rotation_speeds    = rotation_speeds

        # Start on fastest
        self.trans_idx = len(self.translation_speeds)-1
        self.rot_idx   = len(self.rotation_speeds)-1

        self.dead_zone = dead_zone

    def apply_dead_zone(self, value, DEAD_ZONE=0.15):
        if abs(value) < DEAD_ZONE:
            return 0.0
        elif value > 0:
            return (value - DEAD_ZONE) / (1 - DEAD_ZONE)  # Normalize the positive range
        else:
            return (value + DEAD_ZONE) / (1 - DEAD_ZONE)  # Normalize the negative range

    def update_speeds_via_dpad(self):
        """
        Checks the D-pad (hat) state each frame and updates
        the translation/rotation speed indices accordingly.
        """
        # Most controllers only have one hat; index 0
        # dpad_x in [-1, 0, 1], dpad_y in [-1, 0, 1]
        dpad_x, dpad_y = self.joystick.get_hat(0)

        # Left/right changes translation gear
        if dpad_x > 0:  # D-pad right
            self.trans_idx = min(self.trans_idx + 1, len(self.translation_speeds) - 1)
        elif dpad_x < 0:  # D-pad left
            self.trans_idx = max(self.trans_idx - 1, 0)

        # Up/down changes rotation gear
        if dpad_y > 0:  # D-pad up
            self.rot_idx = min(self.rot_idx + 1, len(self.rotation_speeds) - 1)
        elif dpad_y < 0:  # D-pad down
            self.rot_idx = max(self.rot_idx - 1, 0)

    def get_translation_speed(self):
        return self.translation_speeds[self.trans_idx]

    def get_rotation_speed(self):
        return self.rotation_speeds[self.rot_idx]

    def get_action(self):
        """
        Poll the gamepad and construct a 6D action for the robot arm:
          [tx, ty, tz, rx, ry, rz]
        Also returns ( (gripper opened), right_bumper_pressed).
        """
        # Update internal states for the D-pad gear system
        self.update_speeds_via_dpad()

        # Process joystick events so states are updated
        pygame.event.pump()

        # Axes (these indices may vary by controller)
        left_stick_x = self.apply_dead_zone(self.joystick.get_axis(0))   # Left horizontal
        left_stick_y = self.apply_dead_zone(self.joystick.get_axis(1))   # Left vertical
        right_stick_x = self.apply_dead_zone(self.joystick.get_axis(3))  # Right horizontal
        right_stick_y = self.apply_dead_zone(self.joystick.get_axis(4))  # Right vertical
        trigger_l     = self.apply_dead_zone(self.joystick.get_axis(2))  # Left trigger
        trigger_r     = self.apply_dead_zone(self.joystick.get_axis(5))  # Right trigger

        max_speed = self.get_translation_speed()
        rot_speed = self.get_rotation_speed()

        # Basic translation controls
        move_forward_backward = -left_stick_y * max_speed
        move_left_right       =  left_stick_x * max_speed
        move_up_down          = (trigger_r - trigger_l) * max_speed

        # For demonstration: left bumper (LB=4) toggles "yaw" mode
        is_roll_mode = self.joystick.get_button(4)
        if is_roll_mode:
            roll = right_stick_x * rot_speed
            yaw  = 0.0
        else:
            roll = 0.0
            yaw  = -right_stick_x * rot_speed

        pitch = right_stick_y * rot_speed

        # Create the expert action for the 6D arm
        expert_a = np.array([move_forward_backward, 
                             move_left_right, 
                             move_up_down, 
                             roll, 
                             pitch, 
                             yaw], dtype=np.float32)

        # Gripper control with A button (button 0) for closing, x button for opening
        left  = bool(self.joystick.get_button(2))
        right = bool(self.joystick.get_button(0))

        truncated = bool(self.joystick.get_button(3))
        success = bool(self.joystick.get_button(5))

        return expert_a, (left, right, success, truncated)


class GamepadIntervention(gym.ActionWrapper):
    """
    A wrapper that uses a gamepad "expert" to override policy actions
    if the user provides input beyond a small threshold or presses bumpers.

    Includes a gear system for changing translation/rotation speeds with the D-pad.
    """
    def __init__(self, env, action_indices=None, gripper_enabled=True):
        super().__init__(env)

        # Check if environment action space includes a gripper
        self.gripper_enabled = gripper_enabled

        # Create our gamepad "expert" interface
        self.expert = GamepadExpert()

        # Track gripper button presses for the info dict
        self.left, self.right = False, False

        # If you only want to override certain action indices, specify them here
        self.action_indices = action_indices

    def action(self, action: np.ndarray):
        """
        Decide whether to override the incoming 'action' (policy action)
        with the user's gamepad input.
        """
        expert_a, buttons = self.expert.get_action()
        self.left, self.right, self.success, self.truncate = buttons

        # Determine if user is actively intervening
        intervened = False
        # If there's meaningful input on the sticks/triggers, interpret it as user intervention
        if np.linalg.norm(expert_a) > 0.001:
            intervened = True

        if self.gripper_enabled:
            # Use x to open gripper, a to close
            if self.left:  # close gripper
                gripper_action = np.random.uniform(-1, -0.9, size=(1,))
                intervened = True
            elif self.right:  # open gripper
                gripper_action = np.random.uniform(0.9, 1, size=(1,))
                intervened = True
            else:
                gripper_action = np.random.uniform(-0.1, 0.1, size=(1,))
            # Combine arm + gripper
            expert_a = np.concatenate((expert_a, gripper_action), axis=0)
            # Optional random perturbation (comment out if not needed)
            # expert_a[:6] += np.random.uniform(-0.5, 0.5, size=6)

        # If only certain indices should be overwritten:
        if self.action_indices is not None:
            filtered_expert_a = np.zeros_like(expert_a)
            filtered_expert_a[self.action_indices] = expert_a[self.action_indices]
            expert_a = filtered_expert_a

        # If user intervened, return the user (expert) action, else the policy's
        if intervened:
            return expert_a, True
        else:
            return action, False

    def step(self, action):
        """
        Called each environment step. We produce the possibly overridden action.
        """
        new_action, replaced = self.action(action)
        obs, rew, done, truncated, info = self.env.step(new_action)

        # If we replaced the action, record that in info
        if replaced:
            info["intervene_action"] = new_action

        # Store gripper open/close
        info["left"] = self.left
        info["right"] = self.right
        info["success_key"] = self.success
        if self.truncate or truncated:
            truncated=True

        # Store current speeds in the info dict (useful for logging)
        info["translation_speed"] = self.expert.get_translation_speed()
        info["rotation_speed"]    = self.expert.get_rotation_speed()

        return obs, rew, done, truncated, info