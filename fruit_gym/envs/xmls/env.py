import gymnasium as gym
import numpy as np
from mani_skill.agents.base_agent import BaseAgent
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.registration import register_env
from mani_skill.agents.registration import register_agent


@register_agent()
class MyMujocoRobot(BaseAgent):
    uid = "my_mujoco_robot"
    mjcf_path = "/home/emlyn/rl_franka/fruit-gym/fruit_gym/envs/xmls/mjmodel.xml"  # Replace with your MJCF file path
    
    def __init__(self, *args, **kwargs):
        # Set the working directory to the XML file's directory
        # This helps resolve relative paths in the XML file
        import os
        if hasattr(self, 'mjcf_path') and self.mjcf_path != "/home/emlyn/rl_franka/fruit-gym/fruit_gym/envs/xmls/":
            xml_dir = os.path.dirname(os.path.abspath(self.mjcf_path))
            self._original_cwd = os.getcwd()
            os.chdir(xml_dir)
        
        try:
            super().__init__(*args, **kwargs)
        finally:
            # Restore original working directory
            if hasattr(self, '_original_cwd'):
                os.chdir(self._original_cwd)


@register_env("MujocoRobotVisualization-v1")
class MujocoRobotVisualizationEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["my_mujoco_robot"]
    agent: MyMujocoRobot

    def __init__(self, *args, robot_uids="my_mujoco_robot", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        return []

    def _load_scene(self, options):
        """Load the scene with just the robot."""
        pass

    def _initialize_episode(self, env_idx, options):
        """Initialize episode - set robot to default pose."""
        with self._scene.no_rendering():
            # Set robot to a neutral position
            qpos = np.zeros(self.agent.robot.dof)
            self.agent.robot.set_qpos(qpos)
            self.agent.robot.set_qvel(np.zeros(self.agent.robot.dof))

    def _get_obs_extra(self, info):
        """Return empty observation dict."""
        return {}

    def evaluate(self):
        """Dummy evaluation function."""
        return {"success": True}

    def _get_obs_agent(self):
        """Get agent observations."""
        return {
            "qpos": self.agent.robot.get_qpos(),
            "qvel": self.agent.robot.get_qvel(),
        }

    def compute_dense_reward(self, obs, action, info):
        """Dummy reward function."""
        return 0.0

    def compute_normalized_dense_reward(self, obs, action, info):
        """Dummy normalized reward function."""
        return 0.0


def main():
    # Create environment
    env = gym.make(
        "MujocoRobotVisualization-v1",
        obs_mode="none",  # We don't need complex observations for visualization
        render_mode="human",  # Enable visual rendering
        control_mode="pd_joint_pos",  # Use position control
    )
    
    # Reset environment
    obs, info = env.reset()
    
    print("Environment created successfully!")
    print(f"Robot DOF: {env.agent.robot.dof}")
    print("Use this environment to visualize your MuJoCo robot.")
    print("Press Ctrl+C to exit.")
    
    try:
        # Simple visualization loop
        for i in range(1000):
            # Apply zero action (robot stays in place)
            action = np.zeros(env.action_space.shape)
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                obs, info = env.reset()
                
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        env.close()


if __name__ == "__main__":
    main()