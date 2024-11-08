import matplotlib.pyplot as plt
import numpy as np

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