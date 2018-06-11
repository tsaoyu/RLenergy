import gym
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
import numpy as np

from D3HRE.management import Dynamic_environment

class EnergyEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, battery, resource, result_df):
        self.env = Dynamic_environment(battery, resource, None)
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
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """
        ob, reward, done, info = self.env.gym_step(action)
        ob = np.array(ob['usable_capacity'])
        return ob, reward, done, info

    def reset(self):
        self.env.reset()

    def render(self, mode='human', close=False):
        pass

    def def_space(self):
        self.action_space = Discrete(20)
        self.observation_space = Box(low=1,high=1000,shape=(1, ))