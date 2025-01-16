import torch
import gpytorch
import numpy as np
import matplotlib.pyplot as plt

from safe_control_gym.controllers.mpc.flat_gp_utils import GaussianProcess, ConstantMeanAffineGP, ZeroMeanAffineGP
from sklearn.model_selection import train_test_split

import pickle 

seed = 43
output_dir = '/home/tobias/Studium/masterarbeit/code/safe-control-gym/examples/mpc/fgp_output'

# load training data
with open('./examples/mpc/temp-data/gp_train_data.pkl', 'rb') as file:
    data_dict = pickle.load(file)

z_data = data_dict['z']
u_data = data_dict['u']
v_data = data_dict['v']

np_rnd = np.random.default_rng(seed=seed)
noise = np_rnd.normal(0, 0.05, size=v_data.shape)

v_data = v_data+noise

inputs = torch.from_numpy(np.hstack((z_data, u_data)))
targets = torch.from_numpy(np.hstack(v_data[:, 0]))

train_in, test_in, train_tar, test_tar  = train_test_split(inputs, targets, test_size=0.2, random_state=seed)
# train_in = inputs
# train_tar = targets

# Setup GP
gp_type = ZeroMeanAffineGP
likelihood = gpytorch.likelihoods.GaussianLikelihood()
gp = GaussianProcess(gp_type, likelihood, 1, output_dir)

gp.train(train_in, train_tar.squeeze(), n_train=300, learning_rate=0.1) #, n_train=n_train, learning_rate=lr)
#gp.init_with_hyperparam(output_dir)

t = torch.linspace(0, 400, 400)
gp.plot_trained_gp(t) 

means, covs, preds = gp.predict(test_in)
errors = means - test_tar.squeeze()
abs_errors = torch.abs(errors)

indices = np.linspace(0, 100, 1)
plt.figure()
plt.plot(abs_errors)
plt.title("abs_error")

plt.figure()
plt.plot(means)
plt.plot(test_tar)


# plt.show()

# # test with only one query point - like done later in FMPC+SOCP - works like this!

# testPoint = test_in[1,:].unsqueeze(0)
# means, covs, preds = gp.predict(testPoint)
# errors = means - test_tar.squeeze()
# abs_errors = torch.abs(errors)
# gp.model.compute_gammas(test_in)
means_from_gamma, cov_from_gamma, upper_from_gamma, lower_from_gamma  = gp.model.mean_and_cov_from_gammas(test_in)

plt.figure()
plt.plot(means_from_gamma)
plt.plot(means)

plt.figure()
plt.plot(cov_from_gamma)
plt.plot(torch.diag(covs))

plt.figure()
plt.plot(cov_from_gamma-torch.diag(covs))

plt.show()
dummy = 0

