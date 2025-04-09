import numpy as np
import pickle
import matplotlib.pyplot as plt

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

def plot_data_comparison(states, states_ref, time, title, label_x):
    '''plot states 
    input: states/inputs (array: [n_times, n_states])
            time (array: [n_times])'''
    nz = np.shape(states)[1]
    fig, axs = plt.subplots(nz)
    for k in range(nz):
        axs[k].plot(time,states[:, k], color='b', label='actual')
        axs[k].plot(time,states_ref[:, k], color='r', label='reference')        
    axs[0].set_title(title)
    axs[-1].set(xlabel=label_x)
    axs[-1].legend(ncol=3, bbox_transform=fig.transFigure, bbox_to_anchor=(1, 0), loc='lower right')



def evaluateFMPC_SOCP_timing(show_plots = False):

    with open('/home/tobias/Studium/masterarbeit/code/safe-control-gym/examples/mpc/temp-data/fmpc_socp_data_quadrotor_traj_tracking.pkl', 'rb') as file:
        data_dict_fmpc = pickle.load(file)
        metrics_dict = data_dict_fmpc['metrics']
        data_dict_fmpc = data_dict_fmpc['trajs_data']
        
    total_time = np.array(data_dict_fmpc['inference_time_data'])
    total_time = total_time.reshape(-1)
    socp_solve_time = data_dict_fmpc['controller_data'][0]['socp_solve_time'][0]
    gp_time = data_dict_fmpc['controller_data'][0]['gp_time'][0]
    fs_obs_time = data_dict_fmpc['controller_data'][0]['fs_obs_time'][0]
    mpc_time = data_dict_fmpc['controller_data'][0]['mpc_time'][0]
    safety_filt_time = data_dict_fmpc['controller_data'][0]['safety_filt_time'][0]
    dyn_ext_time = data_dict_fmpc['controller_data'][0]['dyn_ext_time'][0]
    


    # print(total_time)


    if show_plots:
        index = range(np.shape(total_time)[0])
        fig, ax = plt.subplots(5, 2)
        ax[0, 0].plot(index, total_time*1000, label='total time')
        # ax[0, 0].plot(index, u_socp[:, 0], label='socp')
        ax[0, 0].set_title('Total execution time select_action')
        ax[0, 0].legend()

        ax[0, 1].plot(index, safety_filt_time*1000, label='safety filter total')
        ax[0, 1].set_title('Total time safety filter')
        ax[0, 1].legend()
  
        ax[1, 0].plot(index, fs_obs_time*1000, label='flat state estimator')
        ax[1, 0].set_title('Flat state estimation')
        # ax[0, 1].set_ylabel('degrees')
        ax[1, 0].legend()

        ax[1, 1].plot(index, gp_time*1000, label='GP time')
        ax[1, 1].set_title('GP inference time')
        # ax[1, 1].set_ylabel('degrees')
        ax[1, 1].legend()

        ax[2, 0].plot(index, mpc_time*1000, label='MPC time')
        # ax[2, 0].plot(index, v_des[:, 0], label='v0 desired')
        ax[2, 0].set_title('MPC execution time')
        ax[2, 0].legend()

        ax[2, 1].plot(index, socp_solve_time*1000, label='SOCP solve')
        ax[2, 1].set_title('SOCP solve time')
        ax[2, 1].legend()

        ax[3, 0].plot(index, dyn_ext_time*1000, label='system extension')
        ax[3, 0].set_title('System extension time')
        ax[3, 0].legend()

        # ax[3, 1].plot(index, total_time, label='Total time measured')
        # ax[3, 1].plot(index, safety_filt_time, label='Safety filter')
        # ax[3, 1].plot(index, safety_filt_time+mpc_time, label='Safety filter + MPC')
        # ax[3, 1].plot(index, safety_filt_time+mpc_time+fs_obs_time+dyn_ext_time, label='Safety + MPC + Ext + Obs')
        # ax[3, 1].set_title('Summary of Control Side')
        ax[4, 0].plot(index, total_time*1000, label='Total time measured')
        ax[4, 0].plot(index, safety_filt_time*1000, label='Safety filter')
        ax[4, 0].plot(index, (safety_filt_time+mpc_time)*1000, label='Safety filter + MPC')
        ax[4, 0].plot(index, (safety_filt_time+mpc_time+fs_obs_time+dyn_ext_time)*1000, label='Safety + MPC + Ext + Obs')
        ax[4, 0].set_title('Summary of Control Side')
        ax[4, 0].legend()

        # ax[4, 0].plot(index, d_slack, label='SOCP Slack')
        # ax[4, 0].set_title('SOCP slack variable stability')

        ax[4, 1].plot(index, safety_filt_time*1000, label='Safety filter total')
        ax[4, 1].plot(index, gp_time*1000, label='GP time')
        ax[4, 1].plot(index, (gp_time + socp_solve_time)*1000, label='GP + SOCP solve')
        ax[4, 1].set_title('Summary Safety Filter')
        ax[4, 1].legend()

        # ax[5, 0].plot(index, cost_val, label='SOCP Cost Total')
        # ax[5, 0].plot(index, cost_val_lin_part, label='SOCP Cost Linear Term')
        # ax[5, 0].plot(index, q_dummy, label='SOCP Cost Quadratic Term')
        # ax[5, 0].plot(index, cost_val_lin_part + q_dummy, label='SOCP Cost Total from comp') # sanity check if it all adds up right
        # ax[5, 0].set_title('SOCP cost')
        # ax[5, 0].legend()

        # ax[5, 1].plot(index, socp_solve_time, label='SOCP')
        # ax[5, 1].plot((0, np.shape(u_analytic_ext)[0]), (np.mean(socp_solve_time), np.mean(socp_solve_time)), label='SOCP mean')
        # ax[5, 1].set_title('Solve times in s')
        # ax[5, 1].legend()

        # ax[6, 0].plot(index, d_slack2, label='SOCP Slack dynExt')
        # ax[6, 0].set_title('SOCP slack variable dynamic extension')

        # ax[6, 1].plot(index, d_slack3, label='SOCP Slack state const')
        # ax[6, 1].set_title('SOCP slack variable state constraint')

        # ax[7, 0].plot(index, thrust_dot, label='Tc_dot')
        # ax[7, 0].set_title('Thrust dot in extension')

        plt.show()




if __name__=="__main__":
    evaluateFMPC_SOCP_timing(show_plots=True)