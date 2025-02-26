from gymnasium.envs.registration import register

register(id="PickStrawbEnv-v1", entry_point="fruit_gym.envs:PickStrawbEnv" , max_episode_steps=1000)
register(id="PickMultiStrawbEnv-v1", entry_point="fruit_gym.envs:PickMultiStrawbEnv" , max_episode_steps=1000)
