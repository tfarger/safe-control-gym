import numpy as np
import pickle

from plottingUtils import *

horizon = 40 # Horizon in FMPC, reference is longer than measurement by this amount

with open('/home/tobias/Studium/masterarbeit/code/safe-control-gym/examples/mpc/temp-data/fmpc_data_quadrotor_traj_tracking.pkl', 'rb') as file:
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

# plot_data_comparison(u, u_ref[0:-horizon], range(np.shape(u)[0]), 'FMPC vs NMPC reference inputs', 'datapoint number')
# plot_data(u-u_ref[:-horizon], range(np.shape(u)[0]), 'Differences on inputs: u - u_ref', 'datapoint number')

# plot_data_comparison(obs_x, x_ref[0:-horizon], range(np.shape(u)[0]), 'FMPC vs NMPC reference states X', 'datapoint number')
# plot_data_comparison(obs_z, z_ref[0:-horizon], range(np.shape(u)[0]), 'FMPC vs NMPC reference flat states Z', 'datapoint number')
# plot_data(obs_z-z_ref[:-horizon], range(np.shape(u)[0]), 'Tracking error on flat states: obs_z - z_ref', 'datapoint number')
# # plot_data((obs_z-z_ref[:-horizon])/z_ref[:-horizon], range(np.shape(u)[0]), 'Relative tracking error on flat states: (obs_z - z_ref)/z_ref', 'datapoint number') # does not show anything good, singularities

# plot_data_comparison(v, v_ref[0:-horizon], range(np.shape(u)[0]), 'FMPC flat inputs V', 'datapoint number')

plt.show()

# compute performance metrics
delta_z = obs_z-z_ref[0:-horizon]

max_error = np.max(np.abs(delta_z), axis=0)
min_error = np.min(np.abs(delta_z), axis=0)

print('State Variable: maximal tracking error')
print('x : {:10.3f}mm'.format(max_error[0]*1000))
print('z : {:10.3f}mm'.format(max_error[4]*1000))
print('-------------------')
print('x_dot : {:10.3f}mm/s'.format(max_error[1]*1000))
print('z_dot : {:10.3f}mm/s'.format(max_error[5]*1000))
