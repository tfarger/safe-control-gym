import numpy as np
import pickle

import matplotlib.pyplot as plt

from safe_control_gym.controllers.mpc.fmpc import _get_z_from_regular_states_2D_att

from examples.mpc.mpc_quad_gp_training_data import run

import os
import yaml

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

def run_sim_scg(additional=''):
    run(gui=False, save_data=True, ALGO='fmpc', ADDITIONAL=additional)

def load_data_scg():
    with open('./temp-gp-training-data/fmpc_data_quadrotor_traj_tracking.pkl', 'rb') as file:
        data_dict = pickle.load(file)
        data_dict = data_dict['trajs_data']        
    states_x = data_dict['state'][0] # Nx6
    actions_u = data_dict['action'][0] # Nx2
    return states_x, actions_u

def postprocess_data(x, u, dt, inertial_prop, g):
    # approximate u_dot with finite differences
    u_dot = np.zeros(np.shape(u))
    for i in range(1, np.shape(u)[0]-1, 1):
        u_dot[i, :] = (-u[i-1, :] + u[i+1, :])/(2*dt)

    # build z vector from this
    z = np.zeros((np.shape(x)[0], 8))
    for i in range(np.shape(x)[0]-1):
        z[i, :] = _get_z_from_regular_states_2D_att(x[i, :].transpose(), u[i, 0], u_dot[i, 0], inertial_prop, g)

    # approximate v flat input = 4th derivative
    # from z: z_dot
    z_dot = np.zeros(np.shape(z))
    for i in range(1, np.shape(z)[0]-1, 1):
        z_dot[i, :] = (-z[i-1, :] + z[i+1, :])/(2*dt)
    v_from_z = np.zeros([np.shape(z)[0], 2])
    v_from_z[:, 0] = z_dot[:, 3]
    v_from_z[:, 1] = z_dot[:, 7]

    # approximate u_ddot for input data with dynamic extension
    u_ddot = np.zeros(np.shape(u))
    for i in range(1, np.shape(u)[0]-1, 1):
        u_ddot[i, :] = (u[i-1, :] - 2*u[i, :] + u[i+1, :])/(dt**2)

    # create input data u_bar for system with dynamic extension 
    # equal to [u_ddot[0], u[1]] = [Tc_ddot, theta_c]
    u_bar = np.zeros(np.shape(u))
    u_bar[:, 0] = u_ddot[:, 0]
    u_bar[:, 1] = u[:, 1]
    
    return z, u_bar, v_from_z

def get_one_dataset(additional, PLOT_RUN, traj_sample_time, inertial_prop, g):
    run_sim_scg(additional)
    states, actions = load_data_scg()
    z, u_bar, v = postprocess_data(states, actions, traj_sample_time, inertial_prop, g)

    # get data from yaml file
    SYS = 'quadrotor_2D_attitude'
    TASK = 'tracking'
    ADDITIONAL = additional
    assert os.path.exists(f'./config_overrides/{SYS}/{SYS}_{TASK}{ADDITIONAL}.yaml'), f'./config_overrides/{SYS}/{SYS}_{TASK}{ADDITIONAL}.yaml does not exist'
    with open(f'./config_overrides/{SYS}/{SYS}_{TASK}{ADDITIONAL}.yaml') as file:
        data_yaml = yaml.safe_load(file)
    episode_len_sec = data_yaml['task_config']['episode_len_sec']
    ctrl_freq = data_yaml['task_config']['ctrl_freq']
    num_cycles = data_yaml['task_config']['task_info']['num_cycles']

    # find start/stop index of cycle 0.5 - 1.5 (full figure8 but starting in the middle)
    use_cycle_num = 1.5
    start_index = int((episode_len_sec/num_cycles) * ctrl_freq *(use_cycle_num-1))
    stop_index = int((episode_len_sec/num_cycles) * ctrl_freq *(use_cycle_num))

    # cut down data
    u_bar_data = u_bar[start_index:stop_index, :]
    z_data = z[start_index:stop_index, :]
    v_data = v[start_index:stop_index, :]

    # assemble into GP training data
    x_train = np.hstack((z_data, u_bar_data))

    if PLOT_RUN:
        # plotting
        times = np.linspace(episode_len_sec/num_cycles*(use_cycle_num-1), episode_len_sec/num_cycles*use_cycle_num, np.shape(u_bar_data)[0])
        plot_data(z_data, times, 'Flat States Z', 'time')
        plot_data(v_data, times, 'Flat Input Trajectory V', 'time')
        plot_data(u_bar_data, times, 'GP training data input u_bar', 'time' )
        plt.show()
    return x_train, v_data


###################################################################################
################# Main part #######################################################
PLOT_RUN = False
ctrl_freq = 50 # of controller in dataset
traj_sample_time = 1/ctrl_freq
additional_list_train = ['_tr2', '_tr3', '_tr4', '_tr5']
additional_list_test = ['_te1', '_te2']

g=9.8

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

# Training data ###################################################################
inputs = []
targets = []

for additional in additional_list_train:
    x_train, v_data = get_one_dataset(additional, PLOT_RUN, traj_sample_time, inertial_prop, g)
    inputs.append(x_train)
    targets.append(v_data)

inputs = np.vstack(inputs)
targets = np.vstack(targets)

indices = np.arange(0, np.shape(inputs)[0])
plot_data(inputs, indices, 'GP training inputs', 'index')
plot_data(targets, indices, 'GP training targets', 'index')
plt.show()


train_data_dict = {'inputs': inputs, 'targets': targets}

with open('./fgp/gp_train_data.pkl', 'wb') as file:
    pickle.dump(train_data_dict, file)

# Test data #######################################################################

inputs = []
targets = []

for additional in additional_list_test:
    x_train, v_data = get_one_dataset(additional, PLOT_RUN, traj_sample_time, inertial_prop, g)
    inputs.append(x_train)
    targets.append(v_data)

inputs = np.vstack(inputs)
targets = np.vstack(targets)

indices = np.arange(0, np.shape(inputs)[0])
plot_data(inputs, indices, 'GP test inputs', 'index')
plot_data(targets, indices, 'GP test targets', 'index')
plt.show()


test_data_dict = {'inputs': inputs, 'targets': targets}

with open('./fgp/gp_test_data.pkl', 'wb') as file:
    pickle.dump(test_data_dict, file)



