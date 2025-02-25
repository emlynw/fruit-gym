from gymnasium.envs.registration import register

register(id="PickStrawbEnv-v1", entry_point="gym_INB0104.envs:PickStrawbEnv" , max_episode_steps=1000)
register(id="PickMultiStrawbEnv-v1", entry_point="gym_INB0104.envs:PickMultiStrawbEnv" , max_episode_steps=1000)
