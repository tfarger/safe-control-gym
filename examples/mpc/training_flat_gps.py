import torch
import gpytorch
import numpy as np
import matplotlib.pyplot as plt

from safe_control_gym.controllers.mpc.flat_gp_utils import GaussianProcess, ConstantMeanAffineGP, ZeroMeanAffineGP
from sklearn.model_selection import train_test_split

import pickle 
import os
import sys

def plot_trained_gp(targets, means, preds, fig_count=0, show=False):
    lower, upper = preds.confidence_region()
    fig_count += 1
    plt.figure(fig_count)
    plt.fill_between(list(range(lower.shape[0])), lower.detach().numpy(), upper.detach().numpy(), alpha=0.5, label='95%')
    plt.plot(means.squeeze(), 'r', label='GP Mean')
    plt.plot(targets.squeeze(), '*k', label='Targets')
    plt.legend()
    plt.title('Fitted GP')
    plt.xlabel('Time (s)')
    plt.ylabel('v')
    if show:
        plt.show()

    return fig_count

# Parameters
seed = 43
output_dir = '/home/tobias/Studium/masterarbeit/code/safe-control-gym/examples/mpc/fgp/gp_v1'

# #debugger
# training_data_file = './examples/mpc/fgp/gp_train_data.pkl'
# eval_data_file = './examples/mpc/fgp/gp_test_data.pkl' # more evaluation data, test it on unseen speeds

#run from folder
training_data_file = './fgp/gp_train_data.pkl'
eval_data_file = './fgp/gp_test_data.pkl' # more evaluation data, test it on unseen speeds

noise_std = 0.03 # for artificial noise

test_size = 0.2 # for train test split

num_datapoints_train = 650 # downsample to this number 

N_train = 400 # number of training iterations in the GP
learning_rate = 0.05

# load data
with open(training_data_file, 'rb') as file:
    train_data = pickle.load(file)

with open(eval_data_file, 'rb') as file:
    eval_data = pickle.load(file)

inputs_train = train_data['inputs']
targets_train = train_data['targets'] 
targets_train = targets_train[:, 0] # later here also for other GP

inputs_eval = eval_data['inputs']
targets_eval = eval_data['targets'] 
targets_eval = targets_eval[:, 0] # later here also for other GP

# add artificial noise
np_rnd = np.random.default_rng(seed=seed)
noise = np_rnd.normal(0, noise_std, size=targets_train.shape)

targets_train = targets_train+noise

# move to Torch
inputs = torch.from_numpy(inputs_train)
targets = torch.from_numpy(targets_train)

# downsampling
interval = int(np.ceil(inputs.shape[0]/num_datapoints_train))
inputs = inputs[::interval, :]
targets = targets[::interval]

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
    gp.train(train_in, train_tar.squeeze(), n_train=N_train, learning_rate=learning_rate)
sys.stdout = orig_stdout

# # plot trained GP, only useful if training data is not shuffled!
# t = torch.linspace(0, 1, train_tar.shape[0])
# gp.plot_trained_gp(t) 

# check on test split
means, covs, preds = gp.predict(test_in)
errors = means - test_tar.squeeze()
abs_errors = torch.abs(errors)
fig, ax = plt.subplots(2)
fig.suptitle('Trained GP evaluated on test split')
ax[0].plot(abs_errors)
ax[0].set_title("abs_error")

ax[1].plot(means, label='predicted means' )
ax[1].plot(test_tar,  label=' test split targets')
ax[1].legend()

print('Test split mean error:', torch.mean(abs_errors).numpy())

# # test implementation of gammas
# means_from_gamma, cov_from_gamma, upper_from_gamma, lower_from_gamma  = gp.model.mean_and_cov_from_gammas(test_in)
# fig, ax = plt.subplots(2, 2)
# ax[0, 0].plot(means_from_gamma, label='mean from gamma')
# ax[0, 0].plot(means, label='means predict')
# ax[0, 0].legend()
# ax[0, 0].set_title('Compare mean predict vs. mean from gamma')

# ax[0, 1].plot(cov_from_gamma, label='covs from gamma')
# ax[0, 1].plot(torch.diag(covs), label='covs predict')
# ax[0, 1].legend()
# ax[0, 1].set_title('Compare cov predict vs. cov from gamma')

# ax[1, 0].plot(means_from_gamma-means)
# ax[1, 0].set_title('Difference in means')

# ax[1, 1].plot(cov_from_gamma-torch.diag(covs))
# ax[1, 1].set_title('Difference in covs')

# Show Quality on unseen data
mean_eval, cov_eval, preds = gp.predict(inputs_eval)
figcount = plot_trained_gp(targets_eval, mean_eval, preds, 2)
errors = mean_eval - targets_eval.squeeze()
abs_errors = torch.abs(errors)
print('Eval set mean error:', torch.mean(abs_errors).numpy())

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
dummy = 0

