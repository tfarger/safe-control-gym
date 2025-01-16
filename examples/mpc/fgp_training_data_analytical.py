import numpy as np
import pickle

import matplotlib.pyplot as plt

from safe_control_gym.controllers.mpc.fmpc import _get_z_from_regular_states_2D_att

"""
Get training data for GP of FMPC_SOCP 
using the exact system model available in FMPC.py
for 2D attitude model
"""
def plot_data(states, time, title, label_x):
    '''plot states 
    input: states/inputs (array: [n_times, n_states])
            time (array: [n_times])'''
    nz = np.shape(states)[1]
    fig, axs = plt.subplots(nz)
    for k in range(nz):
        axs[k].plot(time,states[:, k], color='b', label=' ')
                
    axs[0].set_title(title)
    axs[-1].set(xlabel=label_x)


ctrl_freq = 50 # of controller in dataset
# to cut off transient part if needed
start_index = 70
stop_index = 570

# 2D Quadrotor Attitude model. TODO: Take from env!
inertial_prop = {}
inertial_prop['alpha_1'] = -140.8
inertial_prop['alpha_2'] = -13.4
inertial_prop['alpha_3'] = 124.8
inertial_prop['beta_1'] = 18.11
inertial_prop['beta_2'] = 3.68


# These are the 3D Quad inertial properties!
# inertial_prop = {} # not as a nice variable in the env yet, thats why its defined here again
# inertial_prop['alpha_0'] = 20.907574256269616
# inertial_prop['alpha_1'] = 3.653687545690674
# inertial_prop['beta_0'] = -130.3
# inertial_prop['beta_1'] = -16.33
# inertial_prop['beta_2'] = 119.3
# inertial_prop['gamma_0'] = -99.94
# inertial_prop['gamma_1'] = -13.3
# inertial_prop['gamma_2'] = 84.73
g=9.8
#############################################
traj_sample_time = 1/ctrl_freq

with open('./examples/mpc/temp-data/fmpc_data_quadrotor_traj_tracking.pkl', 'rb') as file:
    data_dict = pickle.load(file)
    data_dict = data_dict['trajs_data']
    
states_all = data_dict['state'][0] # Nx6
actions_all = data_dict['action'][0] # Nx2
#timestamps = data_dict['timestamp'][0] # real time not sim time --> not used

# approximate u_dot with finite differences
u_dot = np.zeros(np.shape(actions_all))
for i in range(1, np.shape(actions_all)[0]-1, 1):
    u_dot[i, :] = (-actions_all[i-1, :] + actions_all[i+1, :])/(2*traj_sample_time)

# build z vector from this
z = np.zeros((np.shape(states_all)[0], 8))
for i in range(np.shape(states_all)[0]-1):
    z[i, :] = _get_z_from_regular_states_2D_att(states_all[i, :].transpose(), actions_all[i, 0], u_dot[i, 0], inertial_prop, g)

# approximate v flat input = 4th derivative
# from z: z_dot
z_dot = np.zeros(np.shape(z))
for i in range(1, np.shape(z)[0]-1, 1):
    z_dot[i, :] = (-z[i-1, :] + z[i+1, :])/(2*traj_sample_time)
v_from_z = np.zeros([np.shape(z)[0], 2])
v_from_z[:, 0] = z_dot[:, 3]
v_from_z[:, 1] = z_dot[:, 7]

# approximate u_ddot for input data with dynamic extension
u_ddot = np.zeros(np.shape(actions_all))
for i in range(1, np.shape(actions_all)[0]-1, 1):
    u_ddot[i, :] = (actions_all[i-1, :] - 2*actions_all[i, :] + actions_all[i+1, :])/(traj_sample_time**2)

# cut off stuff thats not needed
u_data = actions_all[start_index:stop_index, :]
u_dot_data = u_dot[start_index:stop_index, :]
u_ddot_data = u_ddot[start_index:stop_index, :]
z_data = z[start_index:stop_index, :]
v_data = v_from_z[start_index:stop_index, :]

# create input data u_bar for system with dynamic extension 
# equal to [u_ddot[0], u[1]] = [Tc_ddot, theta_c]
u_bar = np.zeros(np.shape(u_data))
u_bar[:, 0] = u_ddot_data[:, 0]
u_bar[:, 1] = u_data[:, 1]

# transpose data if necessary, combine into vectors if necessary 
# TODO
# save data away

if True:
    # plotting
    times = np.linspace(0, 1, np.shape(u_data)[0]) # slightly off by one horizon
    plot_data(z_data, times, 'Flat States Z', 'time')
    plot_data(u_data, times, 'Input Trajectory U', 'time')
    plot_data(v_data, times, 'Flat Input Trajectory V', 'time')
    plot_data(u_ddot_data, times, 'Second Input Derivative U_ddot', 'time')
    plot_data(u_dot_data, times, 'First Input Derivative U_dot', 'time')

    plot_data(u_bar, times, 'GP training data input', 'time' )
    plt.show()

if True:
    gp_data_dict = {'u': u_bar, 'z': z_data, 'v': v_data}

    with open('./examples/mpc/temp-data/gp_train_data.pkl', 'wb') as file:
        pickle.dump(gp_data_dict, file)
