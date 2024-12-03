import numpy as np
import pickle
import matplotlib.pyplot as plt

from plottingUtils import *



def evaluateFMPC(show_plots = False):

    with open('/home/tobias/Studium/masterarbeit/code/safe-control-gym/examples/mpc/temp-data/fmpc_data_quadrotor_traj_tracking.pkl', 'rb') as file:
    # with open('/home/tobias/Studium/masterarbeit/code/safe-control-gym/examples/mpc/temp-data/fmpc_data_quadrotor_stabilization.pkl', 'rb') as file:
        data_dict_fmpc = pickle.load(file)
        metrics_dict = data_dict_fmpc['metrics']
        data_dict_fmpc = data_dict_fmpc['trajs_data']
        

    data_dict_fmpc = data_dict_fmpc['controller_data'][0]

    # load data from FMPC
    obs_x = data_dict_fmpc['obs_x'][0]
    obs_z = data_dict_fmpc['obs_z'][0]
    v = data_dict_fmpc['v'][0]
    u = data_dict_fmpc['u'][0]

    v_horizon = data_dict_fmpc['horizon_v'][0]
    z_horizon = data_dict_fmpc['horizon_z'][0]
   
    data_length = np.shape(obs_z)[0] # use reference up until this point, reference includes an additional horizon +1 as datapoints

    with open('/home/tobias/Studium/masterarbeit/code/safe-control-gym/examples/mpc/temp-data/reference_analytic.pkl', 'rb') as file:
        data_dict = pickle.load(file)
    z_ref = data_dict['z_ref'].transpose()
    v_ref = data_dict['v_ref'].transpose()

    # compute performance metrics
    delta_z = obs_z-z_ref[0:data_length]
    
    delta_rms = np.zeros([np.shape(delta_z)[0], 1])
    for i in range(np.shape(delta_z)[0]):
        delta_rms[i] = np.sqrt(delta_z[i, 0]**2 + delta_z[i, 4]**2 + delta_z[i, 8]**2)

    max_error = np.max(np.abs(delta_z), axis=0)
    max_rms_error = np.max(np.abs(delta_rms))
    mean_rms_error = np.mean(np.abs(delta_rms))

    print('State Variable: tracking error')
    print('rms(mean) : {:10.3f}mm'.format(mean_rms_error*1000))
    print('rms(max) : {:10.3f}mm'.format(max_rms_error*1000))
    print('x   : {:10.3f}mm'.format(max_error[0]*1000))
    print('z   : {:10.3f}mm'.format(max_error[4]*1000))




    if show_plots:
        plot_data_comparison(obs_z, z_ref[0:data_length], range(np.shape(u)[0]), 'FMPC vs analytic reference: flat states Z', 'datapoint number')
        plot_data(delta_z, range(np.shape(u)[0]), 'Tracking error on flat states: obs_z - z_ref', 'datapoint number')
        plot_data_comparison(v, v_ref[0:data_length], range(np.shape(u)[0]), 'FMPC vs analytic reference: flat inputs V', 'datapoint number')
        plot_data(u, range(np.shape(u)[0]), 'Input u to system', 'datapoint number')
        # RMS error
        plt.figure()
        plt.plot(range(np.shape(u)[0]), delta_rms*1000)
        plt.title('RMS position tracking error in mm')

        plt.show()




if __name__=="__main__":
    evaluateFMPC(show_plots=True)