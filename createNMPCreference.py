import numpy as np
import pickle

from safe_control_gym.controllers.mpc.fmpc import _get_total_thrust_dot_from_flat_states, \
    _get_u_from_flat_states, _get_z_from_regular_states, _get_z_from_regular_states_FD, _get_x_from_flat_states

from plottingUtils import *

# parameters of NMPC run
episode_len_sec = 24 # runtime in seconds
num_cycles = 4  # number of circles

ctrl_freq = 50 # of controller
horizon = 40 # FMPC Horizon to exted reference trajectory

# write to file for loading in SCG
CHECK_TRANSFORM = False
SAVE_DATA = True
PLOT_DATA = False
use_cycle_num = 3
#############################################
traj_sample_time = 1/ctrl_freq

with open('./examples/mpc/temp-data/mpc_data_quadrotor_traj_tracking.pkl', 'rb') as file:
    data_dict = pickle.load(file)
    data_dict = data_dict['trajs_data']
    
states_all = data_dict['state'][0] # Nx6
actions_all = data_dict['action'][0] # Nx2
#timestamps = data_dict['timestamp'][0] # real time not sim time --> not used

# approximate u_dot with finite differences
u_dot = np.zeros(np.shape(actions_all))
u_dot_backwards = np.zeros(np.shape(actions_all))
for i in range(1, np.shape(actions_all)[0]-1, 1):
    u_dot[i, :] = (-actions_all[i-1, :] + actions_all[i+1, :])/(2*traj_sample_time)
    u_dot_backwards[i, :] = (-actions_all[i-1, :] + actions_all[i, :])/traj_sample_time

# build z vector from this
z = np.zeros((np.shape(states_all)[0], 8))
for i in range(np.shape(states_all)[0]-1):
    z[i, :] = _get_z_from_regular_states(states_all[i, :].transpose(), actions_all[i, 0], u_dot[i, 0])

# # build z vector with FD as alternative
# z_alt = np.zeros((np.shape(states_all)[0], 8))
# for i in range(1, np.shape(states_all)[0]-1):
#     z_alt[i, :] = _get_z_from_regular_states_FD(states_all[i, :].transpose(), states_all[i-1, :].transpose(), traj_sample_time, actions_all[i, :].transpose(), u_dot[i, 0]+ u_dot[i, 1])

# approximate v flat input = 4th derivative
# from z: z_dot
z_dot = np.zeros(np.shape(z))
for i in range(1, np.shape(z)[0]-1, 1):
    z_dot[i, :] = (-z[i-1, :] + z[i+1, :])/(2*traj_sample_time)
v_from_z = np.zeros([np.shape(z)[0], 2])
v_from_z[:, 0] = z_dot[:, 3]
v_from_z[:, 1] = z_dot[:, 7]
# from x: x_dddot
x_dddot = np.zeros(np.shape(states_all))
for i in range(2, np.shape(states_all)[0]-2, 1):
    x_dddot[i, :] = (-states_all[i-2, :] + 2*states_all[i-1, :] - 2*states_all[i+1, :] + states_all[i+2, :])/(2*traj_sample_time**3)
v_from_x = np.zeros([np.shape(z)[0], 2])
v_from_x[:, 0] = x_dddot[:, 1]
v_from_x[:, 1] = x_dddot[:, 3]


# get the selected circle at the sampling frequency
start_index = int((episode_len_sec/num_cycles) * ctrl_freq *(use_cycle_num-1))
stop_index = int((episode_len_sec/num_cycles) * ctrl_freq *(use_cycle_num)) + horizon

x_traj = states_all[start_index:stop_index, :]
u_traj = actions_all[start_index:stop_index, :]
u_dot_traj = u_dot[start_index:stop_index, :]
z_traj = z[start_index:stop_index, :]
v_traj = v_from_z[start_index:stop_index, :]
v_traj2 = v_from_x[start_index:stop_index, :]

# z_alt_traj = z_alt[start_index:stop_index, :]

# print(states_all[start_index-1])
if CHECK_TRANSFORM:
    # rebuild x vector to check flat transformations
    x_rebuild = np.zeros((np.shape(states_all)[0], 6))
    for i in range(np.shape(states_all)[0]-1):
        x_rebuild[i, :] = _get_x_from_flat_states(z[i, :].transpose())
    x_rebuild_traj = x_rebuild[start_index:stop_index, :]
    times = np.linspace(0, episode_len_sec/num_cycles, np.shape(x_traj)[0]) # slightly off by one horizon
    plot_data_comparison(x_rebuild_traj, x_traj, times, 'Rebuild states x vs. original states x', 'time')
    plot_data(x_rebuild_traj-x_traj, times, 'Difference between x and x_rebuild', 'time')

if PLOT_DATA:
    # plotting
    times = np.linspace(0, episode_len_sec/num_cycles, np.shape(x_traj)[0]) # slightly off by one horizon
    plot_data(z_traj, times, 'Flat States Z', 'time')
    plot_data(u_traj, times, 'Input Trajectory U', 'time')
    plot_data(u_dot_traj, times, 'Input Derivative U_dot', 'time')
    plot_data_comparison(u_dot_traj, u_dot_backwards[start_index:stop_index, :], times, 'Input Derivative: central vs backwards FD', 'time')
    plot_data_comparison(v_traj, v_traj2, times, ' Flat Input Trajectory V: from z vs from x computation', 'time')

    # plot_data_comparison(z_traj, z_alt_traj, times, 'Flat States Z - two ways of computing x/z_ddot', 'time')
    plt.show()
    

if SAVE_DATA:
    reference_dict = {'z_ref': z_traj, 'x_ref': x_traj, 'u_ref': u_traj, 'u_dot_ref': u_dot_traj, 'v_ref':v_traj}

    with open('./examples/mpc/temp-data/reference_NMPC.pkl', 'wb') as file_ref:
        pickle.dump(reference_dict, file_ref)
    
    # print initial values for .yaml file and code
    print('Initial values to copy into SCG:')
    print('x:', x_traj[0])
    print('u', u_traj[0, :])
    print('u_prev', actions_all[start_index-1, :])
    print('u_dot', u_dot_traj[0, :])
