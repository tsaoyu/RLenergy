import gym
import cloudpickle
import numpy as np

from gym.spaces.box import Box
from D3HRE.management import Dynamic_environment

import cloudpickle
import pandas as pd
from bokeh.plotting import figure, show
from bokeh.models import Range1d, LinearAxis
from bokeh.layouts import column
from bokeh.resources import CDN
from bokeh.embed import file_html

def management_result_to_power_report(management_result, result_df):
    """
    Management dataFrame adaptor that concat management result and
    simulation result together.

    :param management_result: dataFrame contains the following fields: SOC,
    Battery, Unmet, Waste and Supply
    :param result_df: dataFrame contains the following fields: Load_demand,
    Prop_load, Hotel_load, solar_power, wind_raw, wind_correction, wind_power
    :return: dataFrame combine those two frames together
    """
    load_resource = result_df[["Load_demand", "Prop_load", "Hotel_load",
               "solar_power", "wind_raw", "wind_correction", "wind_power"]]
    power_report = [management_result, load_resource]
    power_report_df = pd.concat(power_report, axis=1)
    try:
        power_report_df.name = management_result.name
    except AttributeError:
        pass
    return  power_report_df


def bokeh_simple_result_plot(result_df):
    C = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
    fig = figure(plot_width=1400, plot_height=300, x_axis_type='datetime')
    fig.xaxis.axis_label = 'Date'
    fig.yaxis.axis_label = 'Power (generation / demand/ supply) W'
    fig.extra_y_ranges['SOC'] = Range1d(start=0, end=1)
    fig.extra_y_ranges['Reward'] = Range1d(start=-50, end=20)
    fig.add_layout(LinearAxis(y_range_name='SOC', axis_label='SOC'), 'right')
    fig.add_layout(LinearAxis(y_range_name='Reward', axis_label='Reward'), 'right')
    fig.line(result_df.index, result_df['Load_demand'], color=C[0],
             legend=['Demand'])
    fig.line(result_df.index, result_df['Supply'], color=C[1],
             legend=['Supply'])
    fig.line(result_df.index, result_df['Planned'], color=C[2],
             legend=['Planned'])
    fig.line(result_df.index,
             result_df['solar_power'] + result_df['wind_power'],
             color=C[5], legend=['Generation'])
    fig.line(result_df.index, result_df['SOC'], legend='SOC',
             y_range_name='SOC',
             color=C[3], alpha=0.5)
    fig.line(result_df.index, result_df['Reward'], legend='Reward',
             y_range_name='Reward',
             color=C[4], alpha=0.5)
    try:
        fig.title.text = result_df.name
    except  AttributeError:
        pass

    fig.legend.click_policy="hide"
    return fig


# def show_single_bokeh(task_name):
#     prepared_env = task_name + '.pkl'
#     with open(prepared_env, 'rb') as f:
#         env_pack = cloudpickle.load(f)
#     _, task_df, _ = env_pack
#     figs = []
#
#     history = pd.read_pickle('./running_0.pkl'.format(t=task_name))
#     history.name = 'Testing'
#     assembled_df = management_result_to_power_report(history, task_df)
#     figs.append(bokeh_simple_result_plot(assembled_df))
#
#     plot = column(figs)
#     html_output = file_html(plot, CDN, "Example plot")
#     with open('./{t}.html'.format(t=task_name), 'w') as f:
#         f.write(html_output)
#
#     show(plot)


class EnergyEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, task_name, config=None):
        """
        Initialise simulation environment by task name.

        :param task_name: str name of the task e.g. 'Route-3-2013-01-01-3.pkl'
        """
        with open(task_name, 'rb') as f:
            env_pack = cloudpickle.load(f)
        battery, result_df, resource = env_pack

        battery_copy = battery.copy()
        self.env = Dynamic_environment(battery_copy, resource, None, config=config)
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
                whether it'rlenergy time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment'rlenergy last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """
        ob, reward, done, info = self.env.gym_step(action)
        self.done = done
        if isinstance(reward, np.ndarray):
            reward = reward[0][0]
        return ob, reward, done, info

    def reset(self):
        init_state = self.env.reset()
        # Also return initial state
        return init_state

    def render(self, mode='human', close=False):
        if self.done:
            return self.env.simulation_result()
        else:
            pass

    def def_space(self):
        self.action_space = Box(-1., 1., shape=(1,))
        self.observation_space = Box(-1., 1., shape=(4,))

    def scaler(self):
        return self.env.get_scaler()

