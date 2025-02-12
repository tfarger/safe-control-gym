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



def evaluateFMPC_SOCP(show_plots = False):

    with open('/home/tobias/Studium/masterarbeit/code/safe-control-gym/examples/mpc/temp-data/fmpc_socp_data_quadrotor_traj_tracking.pkl', 'rb') as file:
    # with open('/home/tobias/Studium/masterarbeit/code/safe-control-gym/examples/mpc/temp-data/fmpc_data_quadrotor_stabilization.pkl', 'rb') as file:
        data_dict_fmpc = pickle.load(file)
        metrics_dict = data_dict_fmpc['metrics']
        data_dict_fmpc = data_dict_fmpc['trajs_data']
        

    data_dict_fmpc = data_dict_fmpc['controller_data'][0]

    # load data from FMPC
    # u_analytic_noExt = data_dict_fmpc['u_oldFMPC'][0]
    u_analytic_ext = data_dict_fmpc['u_extFT'][0]
    u_socp = data_dict_fmpc['u_extSOCP'][0]
    # u = data_dict_fmpc['u'][0]
    gp_means = data_dict_fmpc['gp_means'][0]
    gp_covs = data_dict_fmpc['gp_covs'][0]





    if show_plots:
        fig, ax = plt.subplots(4, 2)
        ax[0, 0].plot(range(np.shape(u_analytic_ext)[0]), u_analytic_ext[:, 0], label='analytic, dynamic extension')
        ax[0, 0].plot(range(np.shape(u_analytic_ext)[0]), u_socp[:, 0], label='socp')
        ax[0, 0].set_title('Input extended system: Tc_ddot')
        ax[0, 0].legend()

        ax[1, 0].plot(range(np.shape(u_analytic_ext)[0]), (u_analytic_ext[:, 0] - u_socp[:, 0]), label='difference analytic - socp')
        ax[1, 0].set_title('Tc_ddot: Difference analytic - socp')
  
        ax[0, 1].plot(range(np.shape(u_analytic_ext)[0]), u_analytic_ext[:, 1]*180/np.pi, label='analytic, dynamic extension')
        ax[0, 1].plot(range(np.shape(u_analytic_ext)[0]), u_socp[:, 1]*180/np.pi, label='socp')
        ax[0, 1].set_title('Attitude angle theta')
        ax[0, 1].set_ylabel('degrees')
        ax[0, 1].legend()

        ax[1, 1].plot(range(np.shape(u_analytic_ext)[0]), (u_analytic_ext[:, 1] - u_socp[:, 1])*180/np.pi, label='difference analytic - socp')
        ax[1, 1].set_title('Attitude angle theta: Difference analytic - socp')
        ax[1, 1].set_ylabel('degrees')

        ax[2, 0].plot(range(np.shape(u_analytic_ext)[0]), gp_means[:, 0], label='mean0')
        ax[2, 0].set_title('GP0 predictions: mean ')

        ax[2, 1].plot(range(np.shape(u_analytic_ext)[0]), gp_means[:, 1], label='mean1')
        ax[2, 1].set_title('GP1 predictions: mean ')

        ax[3, 0].plot(range(np.shape(u_analytic_ext)[0]), 2* np.sqrt(gp_covs[:, 0]), label='2stddev0')
        ax[3, 0].set_title('GP0 predictions: 2x standard deviation ')

        ax[3, 1].plot(range(np.shape(u_analytic_ext)[0]), 2* np.sqrt(gp_covs[:, 1]), label='2stddev1')
        ax[3, 1].set_title('GP1 predictions: 2x standard deviation ')



        plt.show()




if __name__=="__main__":
    evaluateFMPC_SOCP(show_plots=True)