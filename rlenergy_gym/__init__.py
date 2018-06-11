from gym.envs.registration import register

register(
    id='EnergyEnv-v0',
    entry_point='rlenergy_gym.envs:FooEnv',
)