import numpy as np
import pickle

import matplotlib.pyplot as plt

from safe_control_gym.controllers.mpc.fmpc import _get_z_from_regular_states_2D_att

from examples.mpc.mpc_quad_gp_training_data import run

import os
import yaml

"""
Get training data for GP of FMPC_SOCP 
synthetically by sin and cos functions
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

# for debugging: the transformation that is supposed to be learned with the GP written out analytically
def _get_u_from_flat_states_2D_att_ext(z, v, dyn_pars, g):
    # for system with dynamic extension: u + [Tc_ddot, theta_c]
    beta_1 = dyn_pars['beta_1']
    beta_2 = dyn_pars['beta_2']
    alpha_1 =  dyn_pars['alpha_1']
    alpha_2 =  dyn_pars['alpha_2']
    alpha_3 =  dyn_pars['alpha_3']

    term_acc_sqrd = (z[2])**2 + (z[6]+g)**2 # x_ddot^2 + (z_ddot+g)^2
    theta = np.arctan2(z[2], (z[6]+g))
    theta_dot = (z[3]*(z[6]+g)- z[2]*z[7])/term_acc_sqrd
    theta_ddot = 1/term_acc_sqrd * (v[0]*(z[6]+g) - z[2]*v[1]) + (1/(term_acc_sqrd**2)) * (2*(z[6]+g)*z[7] + 2*z[2]*z[3]) * (z[2]*z[7] - z[3]*(z[6]+g))

    #t = -(beta_2/beta_1) + np.sqrt(term_acc_sqrd)/beta_1
    p = (1/alpha_3) * (theta_ddot - alpha_1*theta -alpha_2*theta_dot)

    t_ddot = 1/beta_1 * 1/np.sqrt(term_acc_sqrd)*((z[3]**2 + z[7]**2 + z[2]*v[0] + (z[6]+g)*v[1]) - ((z[2]*z[3] + (z[6]+g)*z[7])**2)/term_acc_sqrd)
    return np.array([t_ddot, p])


def get_one_dataset(lap_time, PLOT_RUN, traj_sample_time, inertial_prop, g):
    scaling = 1
    z_offset = 1
    times = np.arange(0, lap_time, traj_sample_time)

    z = np.zeros((len(times), 8))
    v = np.zeros((len(times), 2))
    u_bar = np.zeros((len(times), 2))
    for i, t in enumerate(times):
        traj_freq = 2.0 * np.pi / lap_time
        z[i, 0] = scaling * np.sin(traj_freq * t)
        z[i, 1] = scaling * traj_freq * np.cos(traj_freq * t)
        z[i, 2] = -scaling * traj_freq**2 * np.sin(traj_freq * t)
        z[i, 3] = -scaling * traj_freq**3 * np.cos(traj_freq * t)
        z[i, 4] = scaling * np.sin(traj_freq * t) * np.cos(traj_freq * t) + z_offset
        z[i, 5] = scaling * traj_freq * (np.cos(traj_freq * t)**2 - np.sin(traj_freq * t)**2)
        z[i, 6] =  -scaling * traj_freq**2 * 4 * np.sin(traj_freq * t) *np.cos(traj_freq * t) 
        z[i, 7] = scaling * traj_freq**3 * 4 * (np.sin(traj_freq * t)**2 - np.cos(traj_freq*t)**2)  

        v[i, 0] = scaling * traj_freq**4 * np.sin(traj_freq*t)
        v[i, 1] = scaling * traj_freq**4 * 16 * np.sin(traj_freq*t)*np.cos(traj_freq*t)

        u_bar[i, :] = _get_u_from_flat_states_2D_att_ext(z[i, :], v[i, :], inertial_prop, g)
        

    # assemble into GP training data
    x_train = np.hstack((z, u_bar))

    if PLOT_RUN:
        # plotting        
        plot_data(z, times, 'Flat States Z', 'time')
        plot_data(v, times, 'Flat Input Trajectory V', 'time')
        plot_data(u_bar, times, 'GP training data input u_bar', 'time' )
        plt.show()
    return x_train, v


###################################################################################
################# Main part #######################################################
PLOT_RUN = False
ctrl_freq = 100 # of controller in dataset
traj_sample_time = 1/ctrl_freq
lap_time_list_train = [3.25, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8]
lap_time_list_test = [3.75, 5.75, 7.25, 10]

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

for lap_time in lap_time_list_train:
    x_train, v_data = get_one_dataset(lap_time, PLOT_RUN, traj_sample_time, inertial_prop, g)
    inputs.append(x_train)
    targets.append(v_data)

inputs_arr = np.vstack(inputs)
targets_arr = np.vstack(targets)

indices = np.arange(0, np.shape(inputs_arr)[0])
plot_data(inputs_arr, indices, 'GP training inputs z and u', 'index')
plot_data(targets_arr, indices, 'GP training targets v', 'index')
plt.show()


train_data_dict = {'inputs': inputs, 'targets': targets}

with open('./fgp/gp_train_data_fig8.pkl', 'wb') as file:
    pickle.dump(train_data_dict, file)

# Test data #######################################################################

inputs = []
targets = []

for lap_time in lap_time_list_test:
    x_train, v_data = get_one_dataset(lap_time, PLOT_RUN, traj_sample_time, inertial_prop, g)
    inputs.append(x_train)
    targets.append(v_data)

inputs_arr = np.vstack(inputs)
targets_arr = np.vstack(targets)

indices = np.arange(0, np.shape(inputs_arr)[0])
plot_data(inputs_arr, indices, 'GP test inputs', 'index')
plot_data(targets_arr, indices, 'GP test targets', 'index')
plt.show()


test_data_dict = {'inputs': inputs, 'targets': targets}

with open('./fgp/gp_test_data.pkl', 'wb') as file:
    pickle.dump(test_data_dict, file)



