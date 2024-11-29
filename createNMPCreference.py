import numpy as np
import pickle

from safe_control_gym.controllers.mpc.fmpc import _get_total_thrust_dot_from_flat_states, \
    _get_u_from_flat_states, _get_z_from_regular_states, _get_z_from_regular_states_FD, _get_x_from_flat_states

from plottingUtils import *

# parameters of NMPC run
episode_len_sec = 16 # runtime in seconds
num_cycles = 3  # number of circles

ctrl_freq = 60 # of controller
horizon = 40 # FMPC Horizon to exted reference trajectory

# write to file for loading in SCG
CHECK_TRANSFORM = True
SAVE_DATA = True
PLOT_DATA = True
use_cycle_num = 2

inertial_prop = {} # not as a nice variable in the env yet, thats why its defined here again
inertial_prop['alpha_0'] = 20.907574256269616
inertial_prop['alpha_1'] = 3.653687545690674
inertial_prop['beta_0'] = -130.3
inertial_prop['beta_1'] = -16.33
inertial_prop['beta_2'] = 119.3
inertial_prop['gamma_0'] = -99.94
inertial_prop['gamma_1'] = -13.3
inertial_prop['gamma_2'] = 84.73
g=9.8
#############################################
traj_sample_time = 1/ctrl_freq

# with open('./examples/mpc/temp-data/mpc_data_quadrotor_traj_tracking.pkl', 'rb') as file:
with open('./examples/mpc/temp-data/lqr_data_quadrotor_traj_tracking.pkl', 'rb') as file:
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
z = np.zeros((np.shape(states_all)[0], 12))
for i in range(np.shape(states_all)[0]-1):
    z[i, :] = _get_z_from_regular_states(states_all[i, :].transpose(), actions_all[i, 0], u_dot[i, 0], inertial_prop, g)

# # build z vector with FD as alternative
# z_alt = np.zeros((np.shape(states_all)[0], 8))
# for i in range(1, np.shape(states_all)[0]-1):
#     z_alt[i, :] = _get_z_from_regular_states_FD(states_all[i, :].transpose(), states_all[i-1, :].transpose(), traj_sample_time, actions_all[i, :].transpose(), u_dot[i, 0]+ u_dot[i, 1])

# approximate v flat input = 4th derivative
# from z: z_dot
z_dot = np.zeros(np.shape(z))
for i in range(1, np.shape(z)[0]-1, 1):
    z_dot[i, :] = (-z[i-1, :] + z[i+1, :])/(2*traj_sample_time)
v_from_z = np.zeros([np.shape(z)[0], 3])
v_from_z[:, 0] = z_dot[:, 3]
v_from_z[:, 1] = z_dot[:, 7]
v_from_z[:, 2] = z_dot[:, 11]
# from x: x_dddot
x_dddot = np.zeros(np.shape(states_all))
for i in range(2, np.shape(states_all)[0]-2, 1):
    x_dddot[i, :] = (-states_all[i-2, :] + 2*states_all[i-1, :] - 2*states_all[i+1, :] + states_all[i+2, :])/(2*traj_sample_time**3)
v_from_x = np.zeros([np.shape(z)[0], 3])
v_from_x[:, 0] = x_dddot[:, 1]
v_from_x[:, 1] = x_dddot[:, 3]
v_from_x[:, 2] = x_dddot[:, 5]

# get phi_ddot and theta_ddot from FD to check formulas
x_dot = np.zeros(np.shape(states_all))
for i in range(2, np.shape(states_all)[0]-2, 1):
    x_dot[i, :] = (-states_all[i-1, :] + states_all[i+1, :])/(2*traj_sample_time)
angles_ddot_FD = np.zeros([np.shape(z)[0], 2])
angles_ddot_FD[:, 0] = x_dot[:, 8]
angles_ddot_FD[:, 1] = x_dot[:, 9]

def compute_angles_ddot(z, v):
    term_xz_acc_sqrd = (z[2])**2 + (z[10]+g)**2 # x_ddot^2 + (z_ddot+g)^2
    theta_ddot = 1/term_xz_acc_sqrd * (v[0]*(z[10]+g) - z[2]*v[2]) + (1/(term_xz_acc_sqrd**2)) * (2*(z[10]+g)*z[11] + 2*z[2]*z[3]) * (z[2]*z[11] - z[3]*(z[10]+g))
    phi_ddot = ((-((g + z[10])*z[11] + z[2]*z[3])*(((g + z[10])*z[11] + z[2]*z[3])*z[6] - ((g + z[10])**2 + z[2]**2)*z[7])*((g + z[10])**2 + z[2]**2 + z[6]**2) - 2*(((g + z[10])*z[11] + z[2]*z[3])*z[6] - ((g + z[10])**2 + z[2]**2)*z[7])*((g + z[10])**2 + z[2]**2)*((g + z[10])*z[11] + z[2]*z[3] + z[6]*z[7]) + ((g + z[10])**2 + z[2]**2)*(-((g + z[10])*z[11] + z[2]*z[3])*z[7] - ((g + z[10])**2 + z[2]**2)*v[1] + ((g + z[10])*v[2] + z[2]*v[0] + z[3]**2 + z[11]**2)*z[6])*((g + z[10])**2 + z[2]**2 + z[6]**2)))/((((g + z[10])**2 + z[2]**2)**(3/2))*((g + z[10])**2 + z[2]**2 + z[6]**2)**2)
    return np.array([phi_ddot, theta_ddot])

angles_ddot_fromZ = np.zeros((np.shape(states_all)[0], 2))
for i in range(np.shape(states_all)[0]-1):
    angles_ddot_fromZ[i, :] = compute_angles_ddot(z[i, :].transpose(), v_from_z[i,:].transpose())


# get the selected circle at the sampling frequency
start_index = int((episode_len_sec/num_cycles) * ctrl_freq *(use_cycle_num-1))
stop_index = int((episode_len_sec/num_cycles) * ctrl_freq *(use_cycle_num)) + horizon

x_traj = states_all[start_index:stop_index, :]
u_traj = actions_all[start_index:stop_index, :]
u_dot_traj = u_dot[start_index:stop_index, :]
z_traj = z[start_index:stop_index, :]
v_traj = v_from_z[start_index:stop_index, :]
v_traj2 = v_from_x[start_index:stop_index, :]
angles_ddot_FD_traj = angles_ddot_FD[start_index:stop_index, :]
angles_ddot_fromZ_traj = angles_ddot_fromZ[start_index:stop_index, :]

# z_alt_traj = z_alt[start_index:stop_index, :]

# print(states_all[start_index-1])
if CHECK_TRANSFORM:
    # rebuild x vector to check flat transformations
    x_rebuild = np.zeros((np.shape(states_all)[0], 10))
    for i in range(np.shape(states_all)[0]-1):
        x_rebuild[i, :] = _get_x_from_flat_states(z[i, :].transpose(), g)
    x_rebuild_traj = x_rebuild[start_index:stop_index, :]
    times = np.linspace(0, episode_len_sec/num_cycles, np.shape(x_traj)[0]) # slightly off by one horizon
    plot_data_comparison(x_rebuild_traj, x_traj, times, 'Rebuild states x vs. original states x', 'time')
    plot_data(x_rebuild_traj-x_traj, times, 'Difference between x and x_rebuild', 'time')
    plot_data_comparison(angles_ddot_fromZ_traj, angles_ddot_FD_traj, times, 'Phi_ddot and Theta_ddot from states z vs. finite differences on x', 'time')
    plot_data(angles_ddot_fromZ_traj, times, 'angles_ddot from Z', 'time')

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
