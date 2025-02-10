import torch
import gpytorch
import numpy as np
import matplotlib.pyplot as plt

from safe_control_gym.controllers.mpc.flat_gp_utils import GaussianProcess, ConstantMeanAffineGP, ZeroMeanAffineGP
from sklearn.model_selection import train_test_split

from scipy.spatial.distance import pdist, squareform

import pickle 
import os
import sys

from copy import deepcopy

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

def plot_trained_gp(targets, means, preds, fig_count=0, show=False):
    lower, upper = preds.confidence_region()
    fig_count += 1
    # plt.figure(fig_count)
    # plt.fill_between(list(range(lower.shape[0])), lower.detach().numpy(), upper.detach().numpy(), alpha=0.5, label='95%')
    # plt.plot(means.squeeze(), 'r', label='GP Mean')
    # plt.plot(targets.squeeze(), '*k', label='Targets')
    # plt.legend()
    # plt.title('Fitted GP')
    # plt.xlabel('Time (s)')
    # plt.ylabel('v')
    confidence_size = upper.detach().numpy() - means.numpy()

    fig, ax = plt.subplots(2, 1) 
    ax[0].fill_between(list(range(lower.shape[0])), lower.detach().numpy(), upper.detach().numpy(), alpha=0.5, label='95%')
    ax[0].plot(means.squeeze(), 'r', label='GP Mean')
    ax[0].plot(targets.squeeze(), '*k', label='Targets')
    ax[0].legend()
    ax[0].set_title('Fitted GP')
    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('v')

    # ax[1].plot(means.squeeze(), 'r', label='GP Mean') 
    # ax[1].plot(targets.squeeze(), '*k', label='Targets') 
    # ax[1].set_title('GP mean and targets')


    ax[1].set_title('2x standard deviation')
    ax[1].plot(confidence_size) 
    if show:
        plt.show()
    return fig_count
######################################################################################################################
# Parameters
seed = 43
PLOT = True

#debugger
training_data_file = './examples/mpc/fgp/gp_train_data_fig8.pkl'
eval_data_file = './examples/mpc/fgp/gp_test_data.pkl' # more evaluation data, test it on unseen speeds

#run from folder
# training_data_file = './fgp/gp_train_data.pkl'
# eval_data_file = './fgp/gp_test_data.pkl' # more evaluation data, test it on unseen speeds

noise_std_list = [0.2, 1.4] # for artificial noise

test_size = 0.2 # for train test split

N_train = 50000 # number of training iterations in the GP
learning_rate = 0.02

threshold = [0.2, 0.1]
do_gp_nr = 0 # which GP to train, 0 or 1

#############################################################################################################
#### Data preparation 
# load data
with open(training_data_file, 'rb') as file:
    train_data = pickle.load(file)
inputs_train_list = train_data['inputs']
targets_train_list = train_data['targets'] 

# dump first few training datasets
inputs_train_list = inputs_train_list[2:11]
targets_train_list = targets_train_list[2:11]


targets_raw = np.vstack(targets_train_list)
targets_raw_gp = targets_raw[:, do_gp_nr]

inputs_raw = np.vstack(inputs_train_list)

# remove position and velocity data, as the analytic transformation does not depend on it
rows_to_remove = [0, 1, 4, 5]
inputs_raw = np.delete(inputs_raw, rows_to_remove, axis=1)

# check similarity of training data with targets
data_full_gp = np.hstack([inputs_raw, np.expand_dims(targets_raw_gp, 1)])
max_vals = np.max(np.abs(data_full_gp), axis=0)
data_normalized_gp = data_full_gp/max_vals
dist_matrix = squareform(pdist(data_normalized_gp, metric='euclidean'))
plt.figure()
plt.imshow(dist_matrix, cmap='viridis', interpolation='nearest')  # Heatmap
plt.colorbar(label="Distance")  # Add color scale
plt.title("Pairwise Distance of full training data")
plt.xlabel("Point Index")
plt.ylabel("Point Index")



#remove data that is too similar
filtered_indices = []
for i in range(len(data_full_gp)):
    if all(dist_matrix[i, j] >= threshold[do_gp_nr] for j in filtered_indices):
        filtered_indices.append(i)

data_filtered_gp = data_full_gp[filtered_indices, :]
data_filtered_gp_normalized = data_normalized_gp[filtered_indices, :] # just for visualization

# check results of filter
dist_matrix = squareform(pdist(data_filtered_gp_normalized, metric='euclidean'))
plt.figure()
plt.imshow(dist_matrix, cmap='viridis', interpolation='nearest')  # Heatmap
plt.colorbar(label="Distance")  # Add color scale
plt.title(f"Normalized data filtered with distance threshold {threshold[do_gp_nr]}")
plt.xlabel("Point Index")
plt.ylabel("Point Index")

plt.show()

input_data = data_filtered_gp[:, :-1]
targets_gp = data_filtered_gp[:, -1]

np_rnd = np.random.default_rng(seed=seed)
noise = np_rnd.normal(0, noise_std_list[do_gp_nr], size=targets_gp.shape)
targets_gp_noisy = targets_gp + noise

target_data = targets_gp_noisy

# np_rnd = np.random.default_rng(seed=seed)

# targets_noisy_list = [[], []]
# for targets in targets_train_list:
#     for i in [0,1]:
#         # max_val = np.max(targets[:, i])
#         # noise_std = max_val*0.08
#         noise_std = noise_std_list[i]
#         noise = np_rnd.normal(0, noise_std, size=targets[:, i].shape)
#         targets_noisy = targets[:, i] + noise
#         targets_noisy_list[i].append(targets_noisy)


# targets_noisy_gp0 = np.hstack(targets_noisy_list[0])
# targets_noisy_gp1 = np.hstack(targets_noisy_list[1])

# # targets_dwn_gp0 = targets_noisy_gp0[::downsampling_step]
# # targets_dwn_gp1 = targets_noisy_gp1[::downsampling_step]
# targets_dwn_list = [[], []]
# inputs_dwn_list = []
# dwn_sampling_nums = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
# # dwn_sampling_nums = [150, 120, 110, 100, 90, 80, 70, 60, 50, 50, 50]
# # dwn_sampling_nums = [250, 250, 250, 250, 250]
# for i in range(len(inputs_train_list)):
#         index_dwn = np_rnd.choice(len(targets_train_list[i][:, 0]), dwn_sampling_nums[i], replace=False)
#         index_dwn = np.sort(index_dwn)
#         inputs_dwn = inputs_train_list[i][index_dwn, :]
#         inputs_dwn_list.append(inputs_dwn)
#         targets0_dwn = targets_noisy_list[0][i][index_dwn]
#         targets_dwn_list[0].append(targets0_dwn)
#         targets1_dwn = targets_noisy_list[1][i][index_dwn]
#         targets_dwn_list[1].append(targets1_dwn)
# targets_dwn_gp0 = np.hstack(targets_dwn_list[0])
# targets_dwn_gp1 = np.hstack(targets_dwn_list[1])
  

# targets_data = [targets_dwn_gp0, targets_dwn_gp1]

# # downsampling on the input
# # input_data = np.vstack(inputs_train_list)
# # input_data = input_data[::downsampling_step]

# input_data = np.vstack(inputs_dwn_list)

if PLOT:
    # idx_dwn = np.arange(0, len(targets_noisy_gp0), downsampling_step )
    fig, ax = plt.subplots(2)
    ax[0].plot(targets_raw_gp,'.', label='target data')
    ax[0].plot(filtered_indices, targets_gp,'.', label='similar points removed')
    ax[0].plot(filtered_indices, targets_gp_noisy, '.', label='with noise')
    # ax[0].plot(targets_dwn_gp0, '.', label='downsampled')
    ax[0].set_title(f'Training targets GP {do_gp_nr}')
    ax[0].legend()

    # ax[1].plot(targets_raw_gp1, label='target data')
    # ax[1].plot(targets_noisy_gp1, label='with artificial noise')
    # # ax[1].plot(idx_dwn, targets_dwn_gp1, '.',  label='downsampled')
    # ax[1].plot(targets_dwn_gp1, '.',  label='downsampled')
    # ax[1].set_title('Training targets GP1')
    # ax[1].legend()


    t = np.arange(0, np.shape(input_data)[0], 1)
    plot_data(input_data, t, 'Input data, similar points removed', 'index')
    plt.show()


#############################################################################################################
#### GP training and testing

output_dir = f'/home/tobias/Studium/masterarbeit/code/safe-control-gym/examples/mpc/fgp/gp_v{do_gp_nr}'
# Check if the folder exists, and create it if not
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Folder '{output_dir}' created.")

with open(eval_data_file, 'rb') as file:
    eval_data = pickle.load(file)

inputs_train = input_data
targets_train = target_data

inputs_eval = eval_data['inputs']
inputs_eval = np.vstack(inputs_eval)
inputs_eval = np.delete(inputs_eval, rows_to_remove, axis=1)
targets_eval = eval_data['targets'] 
targets_eval = np.vstack(targets_eval)
targets_eval = targets_eval[:, do_gp_nr] 
# move to Torch
inputs = torch.from_numpy(inputs_train)
targets = torch.from_numpy(targets_train)

# train test split
train_in, test_in, train_tar, test_tar  = train_test_split(inputs, targets, test_size=test_size, random_state=seed)
# train_in = inputs
# train_tar = targets

# Setup GP
gp_type = ZeroMeanAffineGP
likelihood = gpytorch.likelihoods.GaussianLikelihood()
gp = GaussianProcess(gp_type, likelihood, 1, output_dir)

fname = os.path.join(output_dir, 'training_output.txt')
orig_stdout = sys.stdout
with open(fname,'w', 1) as print_to_file:
    sys.stdout = print_to_file
    loss_list = gp.train(train_in, train_tar.squeeze(), n_train=N_train, learning_rate=learning_rate)
    # gp.init_with_hyperparam(output_dir)
sys.stdout = orig_stdout

# # plot trained GP, only useful if training data is not shuffled!
# t = torch.linspace(0, 1, train_tar.shape[0])
# gp.plot_trained_gp(t) 

fig, ax = plt.subplots(2)
ax[0].plot(loss_list) #, label='Training')
# ax[0].plot(loss_list_valid, label='Validation')
ax[0].set_title('Loss')
ax[1].plot(loss_list) #, label='Training')
# ax[1].plot(loss_list_valid, label='Validation')
ax[1].set_title('Loss: logarithmic scale')
ax[1].set_xscale('log')

plt.show()

# Print all learned hyperparameters
# print('\n\n**Printing all raw hyperparameters...**\n')
# for name, param in gp.model.named_parameters():
#     param_value = param.detach().numpy()
#     # print(f"{name}: {param_value}")
#     print(f'Parameter name: {name:42} value = {param_value}')

# print('\n\n**Printing all model constraints...**\n')
# for constraint_name, constraint in gp.model.named_constraints():
#     print(f'Constraint name: {constraint_name:55} constraint = {constraint}')

print('\n\n**Printing hyperparameters with constraints...**\n')
print('Likelihood noise')
print(gp.model.likelihood.noise.detach().numpy())
print('kernel variance')
print(gp.model.covar_module.variance.detach().numpy())
print('kernel lengthscale')
print(gp.model.covar_module.length.detach().numpy()[0:4])
print(gp.model.covar_module.length.detach().numpy()[4:8])
print(gp.model.covar_module.length.detach().numpy()[8:12])

# set values to see what happens
# gp.model.likelihood.noise = 0.2

# gp._compute_GP_covariances(train_in) # updating kernel matrix, such that gammas are computed with modified values
# print('\n\n**Printing hyperparameters after modification...**\n')
# print('Likelihood noise')
# print(gp.model.likelihood.noise.detach().numpy())
# print('kernel variance')
# print(gp.model.covar_module.variance.detach().numpy())
# print('kernel lengthscale')
# print(gp.model.covar_module.length.detach().numpy()[0:8])
# print(gp.model.covar_module.length.detach().numpy()[8:16])
# print(gp.model.covar_module.length.detach().numpy()[16:24])

# check on test split
means, covs, preds = gp.predict(test_in)
errors = means - test_tar.squeeze()
abs_errors = torch.abs(errors)
if PLOT:
    fig, ax = plt.subplots(2)
    fig.suptitle('Trained GP evaluated on test split')
    ax[0].plot(abs_errors)
    ax[0].set_title("abs_error")

    ax[1].plot(means, label='predicted means' )
    ax[1].plot(test_tar,  label=' test split targets')
    ax[1].legend()

print('Test split mean error:', torch.mean(abs_errors).numpy())

# test implementation of gammas
means_from_gamma, cov_from_gamma, upper_from_gamma, lower_from_gamma  = gp.model.mean_and_cov_from_gammas(test_in)
if PLOT:
    fig, ax = plt.subplots(2, 2)
    ax[0, 0].plot(means_from_gamma, label='mean from gamma')
    ax[0, 0].plot(means, label='means predict')
    ax[0, 0].legend()
    ax[0, 0].set_title('Compare mean predict vs. mean from gamma')

    ax[0, 1].plot(cov_from_gamma, label='covs from gamma')
    ax[0, 1].plot(torch.diag(covs), label='covs predict')
    ax[0, 1].legend()
    ax[0, 1].set_title('Compare cov predict vs. cov from gamma')

    ax[1, 0].plot(means_from_gamma-means)
    ax[1, 0].set_title('Difference in means')

    ax[1, 1].plot(cov_from_gamma-torch.diag(covs))
    ax[1, 1].set_title('Difference in covs')

# Show Quality on unseen data
mean_eval, cov_eval, preds = gp.predict(inputs_eval)
if PLOT:
    figcount = plot_trained_gp(targets_eval, mean_eval, preds, 3)
errors = mean_eval - targets_eval.squeeze()
abs_errors = torch.abs(errors)
print('Eval set mean error:', torch.mean(abs_errors).numpy())

# Evaluate on unshuffled training data
mean_eval2, cov_eval2, preds2 = gp.predict(inputs)
if PLOT:
    figcount = plot_trained_gp(targets, mean_eval2, preds2, figcount)
errors = mean_eval - targets_eval.squeeze()
abs_errors = torch.abs(errors)
print('Training set mean error:', torch.mean(abs_errors).numpy())

# test reloading from hyperparams: does not seem necessary right now, see FMPC_SOCP: experiments_, line 240

# # test for a single query point, like later in SOCP filter
# point_idx = 9  # select point number 9 randomly, I dont want to use 0
# testPoint = inputs_eval[point_idx,:]
# testPoint = torch.from_numpy(testPoint)
# testPoint = testPoint.unsqueeze(0)
# # mean, cov, preds = gp.predict(testPoint) # this does not work with batch size 1
# gamma1, gamma2, gamma3, gamma4, gamma5 = gp.model.compute_gammas(testPoint)
# gamma1 = gamma1.numpy().squeeze()
# gamma2 = gamma2.numpy().squeeze()
# gamma3 = gamma3.numpy().squeeze()
# gamma4 = gamma4.numpy().squeeze()
# gamma5 = gamma5.numpy().squeeze()

plt.show()
