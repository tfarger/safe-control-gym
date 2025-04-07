# runs experiments for paper
# plots the data for paper
import os
import argparse
from examples.mpc.runs_thesis_analyticFMPC.mpc_experiment_paper import run
import pickle
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

yaml_file_base = './config_overrides_fast/quadrotor_2D_attitude_tracking.yaml'
yaml_file_fmpc_ext = './config_overrides_fast/fmpc_ext_quadrotor_2D_attitude_tracking.yaml'
yaml_file_fmpc = './config_overrides_fast/fmpc_quadrotor_2D_attitude_tracking.yaml'


######### Parameters ###############################
RUN_FMPC_EXT=False
RUN_FMPC=False

GUI = False

ctrl_freq = 50
sample_time = 1/ctrl_freq
num_loops = 2

fig_width = 5.8 # inches
fig_height = 5.8/1.4
plt.rcParams.update({
    'font.size': 10, 
    "text.usetex": True,            # Use LaTeX for text rendering
    "font.family": "serif",         # Match LaTeX font (e.g., Computer Modern)
    "legend.fontsize": 8,           # Legend size
    })
alpha_lines = 0.6


#########################################
data_path_fmpc_ext = './temp-data/fmpc_ext_data_quadrotor_traj_tracking.pkl'
data_path_fmpc = './temp-data/fmpc_data_quadrotor_traj_tracking.pkl'

if RUN_FMPC_EXT:
    if os.path.exists(data_path_fmpc_ext):
        os.remove(data_path_fmpc_ext)
        print(f"{data_path_fmpc_ext} deleted successfully.")
    else:
        print(f"{data_path_fmpc_ext} does not exist.")
    
    # run controller
    run(gui=GUI, save_data=True, algo = 'fmpc_ext', yaml_base = yaml_file_base, yaml_ctrl = yaml_file_fmpc_ext)

if RUN_FMPC:
    if os.path.exists(data_path_fmpc):
        os.remove(data_path_fmpc)
        print(f"{data_path_fmpc} deleted successfully.")
    else:
        print(f"{data_path_fmpc} does not exist.")
    
    # run controller
    run(gui=GUI, save_data=True, algo = 'fmpc', yaml_base = yaml_file_base, yaml_ctrl = yaml_file_fmpc)

#######################################
# extract data
def extract_data(data_file, is_fmpc=False):
    with open(data_file, 'rb') as file:
        data_dict = pickle.load(file)
    metrics = data_dict['metrics']
    traj_data = data_dict['trajs_data']
    states = traj_data['obs'][0]
    # state_mpc = traj_data_mpc['state'][0] # exactly the same as 'obs' for our case (no noise I guess)

    mse_dict = []
    for info in traj_data['info'][0]:
        if 'mse' in info:
            mse_dict.append(info.get('mse'))
        else:
            mse_dict.append(0) # only occurs at initial timestep as far as I can tell
    error = np.array(mse_dict)
    rmse = metrics['rmse']
    # rmse_from_error = np.sqrt(np.mean(error_mpc))
    # rmse_diff = rmse_mpc - rmse_from_error # sanity check: order of e-5 NOTE: rounding errors?

    inference_time = np.array(traj_data['inference_time_data'])
    # diff_time = np.mean(inference_time_mpc) - metrics_mpc['avarage_inference_time'] # = 0, so fine
    
    if is_fmpc:
        state_ref = np.array(traj_data['controller_data'][0]['z_ref'])
    else: 
        state_ref = 0

    # inputs
    action = traj_data['action'][0]

    return states, error, inference_time, rmse, state_ref, action

state_x_fmpc_ext, error_fmpc_ext, inf_time_fmpc_ext, rmse_fmpc_ext, state_ref_fmpc_ext, action_fmpc_ext= extract_data(data_path_fmpc_ext) 
state_x_fmpc, error_fmpc, inf_time_fmpc, rmse_fmpc, state_ref_fmpc, action_fmpc = extract_data(data_path_fmpc, is_fmpc=True)   

# print(np.shape(state_ref_fmpc))
# print(state_ref_fmpc)
# exit()



###########################################################
# data visualization
# Define TUM colors
tum_blue = '#0065BD'
tum_blue_1 = '#98C6EA' # lightest
tum_blue_2 = '#64A0C8'
tum_blue_3 = '#0073CF'
tum_blue_4 = '#005293'
tum_blue_5 = '#003359' # darkest
# accent colors
tum_green = '#A2AD00'
tum_orange = '#E37222'
tum_ivory = '#DAD7CB'
# diagram colors
tum_dia_violet = '#69085A'
tum_dia_dark_blue = '#0F1B5F'
tum_dia_turquoise = '#00778A'
tum_dia_dark_green = '#007C30'
tum_dia_light_green = '#679A1D'
tum_dia_light_yellow = '#FFDC00'
tum_dia_dark_yellow = '#F9BA00'
tum_dia_dark_orange = '#D64C13'
tum_dia_red = '#C4071B'
tum_dia_dark_red = '#9C0D16'

# define color and labels
ref_color = 'black'
fmpc_ext_color = tum_dia_dark_green
fmpc_color = tum_dia_violet

ref_label = '_nolegend_' #'reference'
fmpc_ext_label = 'FMPC Feedback'
fmpc_label = 'FMPC Feedforward'

linewidth = 2.5

limits_x = [-1.1, 1.1]
limits_y = [0.4, 1.6]

# plot of figure 8 in 2D space
plt.figure(figsize=(fig_width, fig_height))
plt.plot(state_ref_fmpc[0, :301, 0], state_ref_fmpc[0, :301, 4], linestyle = 'dashed', color=ref_color, label=ref_label, linewidth=linewidth, alpha=alpha_lines) 
plt.plot(state_x_fmpc_ext[:, 0], state_x_fmpc_ext[:, 2], color=fmpc_ext_color, label=fmpc_ext_label, linewidth=linewidth, alpha=alpha_lines)
plt.plot(state_x_fmpc[:, 0], state_x_fmpc[:, 2], color=fmpc_color, label=fmpc_label, linewidth=linewidth, alpha=alpha_lines)
plt.legend()
plt.xlabel(r'Position x (m)')
plt.ylabel(r'Position z (m)')
plt.xlim(limits_x)
plt.ylim(limits_y)
plt.grid()
plt.savefig("./plots/fig8.pdf", format="pdf", bbox_inches=None)

# plot errors over time
time = np.arange(0, np.shape(error_fmpc)[0]*sample_time, sample_time )
plt.figure(figsize=(fig_width, fig_height))
plt.plot(time, np.sqrt(error_fmpc_ext), color=fmpc_ext_color, label=fmpc_ext_label, linewidth=linewidth, alpha=alpha_lines)
plt.plot(time, np.sqrt(error_fmpc), color=fmpc_color, label=fmpc_label, linewidth=linewidth, alpha=alpha_lines)
plt.legend()
plt.xlabel(r'Time (s)')
plt.ylabel(r'Tracking error (m)')
plt.grid()
plt.savefig("./plots/tracking_error.pdf", format="pdf", bbox_inches=None)

# generate a bunch of metrics on tracking error
# end_idx_first_loop = int(np.shape(state_fmpc)[0]/num_loops)
# def compute_tracking_error(error, end_idx_first_loop):
#     mean_track_err = np.mean(np.sqrt(error))
#     loop1_track_err = np.mean(np.sqrt(error[:end_idx_first_loop]))
#     loop2_track_err = np.mean(np.sqrt(error[end_idx_first_loop:]))
#     return mean_track_err, loop1_track_err, loop2_track_err
# mean_track_err_mpc, loop1_track_err_mpc, loop2_track_err_mpc = compute_tracking_error(error_mpc, end_idx_first_loop)
# mean_track_err_fmpc, loop1_track_err_fmpc, loop2_track_err_fmpc = compute_tracking_error(error_fmpc_ext, end_idx_first_loop)
# mean_track_err_fmpc_socp, loop1_track_err_fmpc_socp, loop2_track_err_fmpc_socp = compute_tracking_error(error_fmpc, end_idx_first_loop)

# print('\nTracking Error: mean(sqrt(sum of squares at each timestep))')
# print('                     NMPC   |  FMPC   | FMPC+SOCP')
# print(' average track_err: {:.2f}mm | {:.2f}mm | {:.2f}mm'.format(mean_track_err_mpc*1000, mean_track_err_fmpc*1000, mean_track_err_fmpc_socp*1000))
# print('1st loop track_err: {:.2f}mm | {:.2f}mm | {:.2f}mm'.format(loop1_track_err_mpc*1000, loop1_track_err_fmpc*1000, loop1_track_err_fmpc_socp*1000))
# print('2nd loop track_err: {:.2f}mm | {:.2f}mm | {:.2f}mm'.format(loop2_track_err_mpc*1000, loop2_track_err_fmpc*1000, loop2_track_err_fmpc_socp*1000))

# print('\nRMSE: sqrt(mean(sum of squares at each timestep))')
# print('                     NMPC   |  FMPC   | FMPC+SOCP')
# print('      average RMSE: {:.2f}mm | {:.2f}mm | {:.2f}mm'.format(rmse_mpc*1000, rmse_fmpc_ext*1000, rmse_fmpc*1000))

##################################################################################
# Inputs
# limits_y1 = [0.2, 0.6]
time = np.arange(0, np.shape(action_fmpc)[0]*sample_time, sample_time )
fig, ax = plt.subplots(2, figsize=(fig_width, fig_height))    
ax[0].plot(time, action_fmpc_ext[:, 0], color=fmpc_ext_color, label=fmpc_ext_label, linewidth=linewidth, alpha=alpha_lines)
ax[0].plot(time, action_fmpc[:, 0], color=fmpc_color, label=fmpc_label, linewidth=linewidth, alpha=alpha_lines)
ax[0].set_ylabel(r'Thrust $T_c$ (N)')
# ax[0].set_ylim(limits_y1)
ax[0].grid()
ax[1].plot(time, action_fmpc_ext[:, 1], color=fmpc_ext_color, label=fmpc_ext_label, linewidth=linewidth, alpha=alpha_lines)
ax[1].plot(time, action_fmpc[:, 1], color=fmpc_color, label=fmpc_label, linewidth=linewidth, alpha=alpha_lines)
ax[1].set_ylabel(r'Angle $\theta_c$ (rad)')
ax[1].grid()
ax[1].set_xlabel(r'Time (s)')
ax[1].legend(loc="upper right")

plt.savefig("./plots/inputs.pdf", format="pdf", bbox_inches=None)

plt.show()
