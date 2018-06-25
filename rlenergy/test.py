import gym
from rlenergy_gym.envs import rl_energy_env

import cloudpickle


with open('prepared_env.pkl', 'rb') as f:
    env_pack = cloudpickle.load(f)

battery, result_df, resource = env_pack
battery_copy = battery.copy()



env = rl_energy_env.EnergyEnv(battery_copy, resource, result_df)
