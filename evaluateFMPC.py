import numpy as np
import pickle

from plottingUtils import *

horizon = 80 # Horizon in NMPC, reference is longer than measurement by this amount

with open('./examples/mpc/temp-data/fmpc_data_quadrotor_traj_tracking.pkl', 'rb') as file:
    data_dict_fmpc = pickle.load(file)
    data_dict_fmpc = data_dict_fmpc['trajs_data']

data_dict_fmpc = data_dict_fmpc['controller_data'][0]

# load data from FMPC
obs_x = data_dict_fmpc['obs_x'][0]
obs_z = data_dict_fmpc['obs_z'][0]
v = data_dict_fmpc['v'][0]
u = data_dict_fmpc['u'][0]
T_dot = data_dict_fmpc['T_dot'][0]
u_ref_f = data_dict_fmpc['u_ref'][0]
v_horizon = data_dict_fmpc['horizon_v'][0]
z_horizon = data_dict_fmpc['horizon_z'][0]
u_horizon = data_dict_fmpc['horizon_u'][0]


with open('/home/tobias/Studium/masterarbeit/code/safe-control-gym/examples/mpc/temp-data/reference_NMPC.pkl', 'rb') as file:
    data_dict = pickle.load(file)
z_ref = data_dict['z_ref']
x_ref = data_dict['x_ref']
u_ref = data_dict['u_ref']
u_dot_ref = data_dict['u_dot_ref']
v_ref = data_dict['v_ref']

plot_data_comparison(u, u_ref[0:-horizon], range(np.shape(u)[0]), 'FMPC vs NMPC reference inputs', 'datapoint number')
# plot_data(u, range(np.shape(u)[0]), 'FMPC inputs', 'datapoint number')

plot_data_comparison(obs_x, x_ref[0:-horizon], range(np.shape(u)[0]), 'FMPC vs NMPC reference states X', 'datapoint number')
plot_data_comparison(obs_z, z_ref[0:-horizon], range(np.shape(u)[0]), 'FMPC vs NMPC reference flat states Z', 'datapoint number')

plot_data_comparison(v, v_ref[0:-horizon], range(np.shape(u)[0]), 'FMPC flat inputs V', 'datapoint number')

plt.show()

