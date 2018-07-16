import torch
import torch.nn as nn
from models.utilities_torch import set_init, v_wrap
import torch.nn.functional as F
import math

from bokeh.plotting import figure, show
from bokeh.models import Range1d, LinearAxis

import pandas as pd


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
    fig = figure(plot_width=1000, plot_height=300, x_axis_type='datetime')
    fig.xaxis.axis_label = 'Date'
    fig.yaxis.axis_label = 'Power (generation / demand/ supply) W'
    fig.extra_y_ranges['SOC'] = Range1d(start=0, end=1)
    fig.add_layout(LinearAxis(y_range_name='SOC', axis_label='SOC'), 'right')
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
    try:
        fig.title.text = result_df.name
    except  AttributeError:
        pass

    fig.legend.click_policy="hide"
    return fig



class Net(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Net, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.a1 = nn.Linear(s_dim, 256)
        self.mu = nn.Linear(256, a_dim)
        self.sigma = nn.Linear(256, a_dim)
        self.c1 = nn.Linear(s_dim, 256)
        self.v = nn.Linear(256, 1)
        set_init([self.a1, self.mu, self.sigma, self.c1, self.v])
        self.distribution = torch.distributions.Normal

    def forward(self, x):
        a1 = F.relu(self.a1(x))
        mu = F.tanh(self.mu(a1))
        sigma = F.softplus(self.sigma(a1)) + 0.001      # avoid 0
        c1 = F.relu(self.c1(x))
        values = self.v(c1)
        return mu, sigma, values

    def choose_action(self, s):
        self.training = False
        mu, sigma, _ = self.forward(s)
        m = self.distribution(mu.view(1, ).data, sigma.view(1, ).data)
        return m.sample().numpy()

    def loss_func(self, s, a, v_t):

        mu, sigma, values = self.forward(s)
        td = v_t - values
        c_loss = td.pow(2)

        m = self.distribution(mu, sigma)
        log_prob = m.log_prob(a)
        entropy = 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(m.scale)  # exploration
        exp_v = log_prob * td.detach() + 0.005 * entropy
        a_loss = -exp_v
        total_loss = (a_loss + c_loss).mean()
        return total_loss

if __name__ == "__main__":
    net = torch.load('110791.net')
    print(net)
    import cloudpickle
    from rlenergy_gym.envs import rl_energy_env

    with open('prepared_env.pkl', 'rb') as f:
        env_pack = cloudpickle.load(f)
    battery, result_df, resource = env_pack
    battery_copy = battery.copy()
    env = rl_energy_env.EnergyEnv(battery_copy, resource, result_df)
    MAX_STEP = len(resource)
    s = env.reset()
    r_total = 0
    for t in range(MAX_STEP):
        a = net.choose_action(v_wrap(s[None, :]))
        # print(s[3] , a)
        s_, r, done, _ = env.step(a.clip(-1, 1))
        s = s_
        r_total += r

    print(r_total)

    assembled_df = management_result_to_power_report(env.render(), result_df)
    show(bokeh_simple_result_plot(assembled_df))
    # print(env.render())

