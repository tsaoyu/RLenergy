from rlenergy.energy_env import EnergyEnv
from gym.envs.registration import register

register(
    id='EnergyEnv-v1',
    entry_point='rlenergy:EnergyEnv',
)
