from envs.energy_env import EnergyEnv
from gym.envs.registration import register

register(
    id='EnergyEnv-v0',
    entry_point='envs.EnergyEnv',
)
