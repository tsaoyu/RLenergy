import gym
import cloudpickle
import numpy as np

from gym.spaces.box import Box
from D3HRE.management import Dynamic_environment


class EnergyEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, task_name):
        """
        Initialise simulation environment by task name.

        :param task_name: str name of the task e.g. 'Route-3-2013-01-01-3.pkl'
        """
        with open(task_name, 'rb') as f:
            env_pack = cloudpickle.load(f)
        battery, result_df, resource = env_pack

        battery_copy = battery.copy()
        self.env = Dynamic_environment(battery_copy, resource, None)
        self.resource = resource
        self.env.set_demand(result_df)
        self.def_space()

    def step(self, action):
        """

        Parameters
        ----------
        action :

        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it'envs time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment'envs last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """
        ob, reward, done, info = self.env.gym_step(action)
        if isinstance(reward, np.ndarray):
            reward = reward[0][0]
        return ob, reward, done, info

    def reset(self):
        init_state = self.env.reset()
        # Also return initial state
        return init_state

    def render(self, mode='human', close=False):
        return self.env.simulation_result()

    def def_space(self):
        self.action_space = Box(-1., 1, shape=(1,))
        self.observation_space = Box(-1, 1, shape=(4,))

    def scaler(self):
        return self.env.get_scaler()