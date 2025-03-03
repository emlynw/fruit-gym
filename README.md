# Fruit Gym

Fruit Gym is a collection of Gymnasium environments for robotic manipulation tasks—specifically, for picking strawberries using a Franka Panda robot. These environments support domain randomization, image observations, and multi-target grasping. They are designed to challenge reinforcement learning algorithms in continuous control and multi-object manipulation tasks.

## Environments

### PickStrawbEnv
**PickStrawbEnv** is a single-target environment where the robot must reach for and grasp one red strawberry among multiple distractor (green) strawberries.

- **Action Space:**  
  A continuous `Box` of shape `(ee_dof + 1,)` (default: `(7,)` when `ee_dof = 6`), with each action component in the range `[-1, 1]`.  
  An action is represented as:  
  `[dz, dy, dx, droll, dpitch, dyaw, dgrasp]`  
  where the displacements and rotations are in the end-effector frame (with z and x swapped for intuitive first-person control) and the last element is the grasp command.

### PickMultiStrawbEnv
**PickMultiStrawbEnv** extends the single-target version to allow multiple red strawberry targets. When the robot successfully grasps a strawberry (both gripper fingers contact a target stem), that strawberry is removed from the environment (its associated geometries are made invisible and its index is removed from the active target lists).

The action and observation spaces, as well as reward components, are similar to **PickStrawbEnv** but are adapted to handle multiple targets.

## Installation

To install Fruit Gym, clone the repository and install it in editable mode:

```bash
git clone https://github.com/emlynw/fruit-gym.git
cd fruit-gym
pip install -e .
```

## Teleoperation
The cameras mounted on the gripper enable efficient first person view teleoperation. Examples of control using a mouse and keyboard or a gamepad are given in fruit_gym/test

## Domain Randomization
Fruit Gym uses a YAML configuration file to control aspects of domain randomization, the yaml file is found in fruit_gym/configs

## Credits

This repository builds upon the work of [serl](https://github.com/rail-berkeley/serl). Thanks to all contributors to the underlying frameworks.

## Citation

If you find Fruit Gym useful in your research, please consider citing it as follows:

To do


