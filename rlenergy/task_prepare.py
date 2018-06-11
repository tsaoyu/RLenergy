from PyResis import propulsion_power
from D3HRE.simulation import Task, Reactive_simulation
from D3HRE.core.mission_utility import Mission
from D3HRE.core.file_reading_utility import read_route_from_gpx
from D3HRE.optimization import Constraint_mixed_objective_optimisation
from D3HRE.core.battery_models import Battery_managed
import numpy as np
import cloudpickle
import pickle

ship = propulsion_power.Ship()
ship.dimension(5.72, 0.248, 0.76, 1.2, 5.72/(0.549)**(1/3), 0.613)

power_consumption_list = {'single_board_computer': {'power': [2, 10], 'duty_cycle': 0.5},
                              'webcam': {'power': [0.6], 'duty_cycle': 1},
                              'gps': {'power': [0.04, 0.4], 'duty_cycle': 0.9},
                              'imu': {'power': [0.67, 1.1], 'duty_cycle': 0.9},
                              'sonar': {'power': [0.5, 50, 0.2], 'duty_cycle': 0.5},
                              'ph_sensor': {'power': [0.08, 0.1], 'duty_cycle': 0.95},
                              'temp_sensor': {'power': [0.04], 'duty_cycle': 1},
                              'wind_sensor': {'power': [0.67, 1.1], 'duty_cycle': 0.5},
                              'servo_motors': {'power': [0.4, 1.35], 'duty_cycle': 0.5},
                              'radio_transmitter': {'power': [0.5, 20], 'duty_cycle': 0.2}}


config = {'load': {'prop_load': {'prop_eff': 0.7,
   'sea_margin': 0.2,
   'temperature': 25}},
 'optimization': {'constraints': {'turbine_diameter_ratio': 1.2,
   'volume_factor': 0.1,
   'water_plane_coff': 0.88},
  'cost': {'battery': 1, 'lpsp': 10000, 'solar': 210, 'wind': 320},
  'method': {'nsga': {'cr': 0.95, 'eta_c': 10, 'eta_m': 50, 'm': 0.01},
   'pso': {'eta1': 2.05,
    'eta2': 2.05,
    'generation': 100,
    'max_vel': 0.5,
    'neighb_param': 4,
    'neighb_type': 2,
    'omega': 0.7298,
    'population': 100,
    'variant': 5}},
  'safe_factor': 0.2},
 'simulation': {'battery': {'B0': 1,
   'DOD': 0.9,
   'SED': 500,
   'eta_in': 0.9,
   'eta_out': 0.8,
   'sigma': 0.005},
  'coupling': {'eff': 0.05}},
 'source': {'solar': {'brl_parameters': {'a0': -5.32,
    'a1': 7.28,
    'b1': -0.03,
    'b2': -0.0047,
    'b3': 1.72,
    'b4': 1.08}}},
 'transducer': {'solar': {'azim': 0,
   'eff': {'k_1': -0.017162,
    'k_2': -0.040289,
    'k_3': -0.004681,
    'k_4': 0.000148,
    'k_5': 0.000169,
    'k_6': 5e-06},
   'loss': 0.1,
   'power_density': 140,
   'tacking': 0,
   'tech': 'csi',
   'tilt': 0},
  'wind': {'power_coef': 0.3,
   'thurse_coef': 0.6,
   'v_in': 2,
   'v_off': 45,
   'v_rate': 15}}}

route_index = 3
START_DATE = '2014-01-01'
SPEED = 3

all_routes = read_route_from_gpx('/home/tony/D3HRE_notebooks/Example data/routes.gpx')
ROUTE = np.array(all_routes[route_index])

mission = Mission(START_DATE, ROUTE, SPEED)
study_task= Task(mission, ship, power_consumption_list)

con_mix_opt = Constraint_mixed_objective_optimisation(study_task, config=config)
champion, champion_x = con_mix_opt.run()
solar_area, wind_area, battery_capacity = champion_x
battery_capacity = battery_capacity * 2
battery = Battery_managed(battery_capacity, config=config)


rea_sim = Reactive_simulation(study_task, config=config)
result_df = rea_sim.result(solar_area, wind_area, battery_capacity)
resource = (result_df.wind_power + result_df.solar_power)



with open('prepared_env.pkl', 'wb') as f:
    cloudpickle.dump([battery, result_df, resource], f)
