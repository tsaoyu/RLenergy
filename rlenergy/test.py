import gym
from baselines import deepq
from rlenergy_gym.envs import rl_energy_env

import cloudpickle
import pickle

from D3HRE.core.battery_models import Battery_managed

with open('prepared_env.pkl', 'rb') as f:
    env_pack = cloudpickle.load(f)

battery, result_df, resource = env_pack
battery_copy = Battery_managed(battery.capacity)



env = rl_energy_env.EnergyEnv(battery_copy, resource, result_df)
model = deepq.models.mlp([64])


act = deepq.learn(
        env,
        q_func=model,
        lr=1e-3,
        max_timesteps=100000,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        print_freq=10,

    )