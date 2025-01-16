import torch
import gpytorch
import numpy as np
import matplotlib.pyplot as plt

from safe_control_gym.controllers.mpc.flat_gp_utils import GaussianProcess, ConstantMeanAffineGP, ZeroMeanAffineGP
from sklearn.model_selection import train_test_split

import pickle 
# N = 200
# x_max = 2
# x_min = 0
# x_delta = x_max*0.1
# # Make training data
# train_z = torch.linspace(x_min, x_max, N).double()
# train_u = torch.linspace(x_min, x_max,N).double()
# train_y = torch.sin(train_z * (2 * np.pi)) + train_u + torch.randn(train_z.size()) * np.sqrt(0.04)
# train_x = torch.stack((train_z, train_u)).T
# # Define Model and train
# output_dir = '/home/tobias/Studium/masterarbeit/code/safe-control-gym/examples/mpc/fgp_output'
# gp_type = ZeroMeanAffineGP
# #gp_type = ConstantMeanAffineGP
# likelihood = gpytorch.likelihoods.GaussianLikelihood()
# gp = GaussianProcess(gp_type, likelihood, 1, output_dir)
# gp.train(train_x, train_y)
# test_x = torch.linspace(x_min-x_delta, x_max+x_delta,int(N/2)).double()
# test_x = torch.stack((test_x,test_x)).T
# # Plot
# means, covs, predictions = gp.predict(test_x)
# lower, upper = predictions.confidence_region()
# # Compute mean and covariance using gammas
# #means_from_gamma = torch.zeros((N,1))
# #covs_from_gamma = torch.zeros((N,1))
# #cov_2_from_gamma = torch.zeros((N,1))
# x = test_x[:,None,0]
# u = test_x[:,None,1]
# #gamma_1, gamma_2, gamma_3, gamma_4, gamma_5, cov_2 = gp.model.compute_gammas(test_x)
# #means_from_gamma = gamma_1 + gamma_2.mul(u)
# #covs_from_gamma = gamma_3 + gamma_4.mul(u) + gamma_5.mul(u**2) + gp.likelihood.noise.detach()
# #cov_2_from_gamma = cov_2
# #upper_from_gamma = means_from_gamma + covs_from_gamma.sqrt()*2
# #lower_from_gamma = means_from_gamma - covs_from_gamma.sqrt()*2
# #upper_from_gamma = means_from_gamma + cov_2_from_gamma.sqrt()*2
# #lower_from_gamma = means_from_gamma - cov_2_from_gamma.sqrt()*2
# means_from_gamma, cov_from_gamma, upper_from_gamma, lower_from_gamma = gp.model.mean_and_cov_from_gammas(test_x)
# # plot gamma computed means
# plt.fill_between(test_x.numpy()[:,0], lower.numpy(), upper.numpy(), alpha=0.5, label='95%')
# plt.fill_between(test_x.numpy()[:,0], lower_from_gamma[:,0].numpy(), upper_from_gamma[:,0].numpy(), alpha=0.5, label='95% from gammas')
# plt.plot(test_x.numpy()[:,0],means,'r', label='GP Mean')
# plt.plot(train_x.numpy()[:,0],train_y,'*k', label='Data')
# plt.plot(test_x.numpy()[:,0],means_from_gamma.detach().numpy()[:,0],'m', label='gamma mean')
# plt.legend()
# plt.show()
# gg = gp.predict(test_x[5,None,:])


# testing with actual data, dim z = 8, dim u = 2

seed = 43
output_dir = '/home/tobias/Studium/masterarbeit/code/safe-control-gym/examples/mpc/fgp_output'

# load training data
with open('./examples/mpc/temp-data/gp_train_data.pkl', 'rb') as file:
    data_dict = pickle.load(file)

z_data = data_dict['z']
u_data = data_dict['u']
v_data = data_dict['v']

inputs = torch.from_numpy(np.hstack((z_data, u_data)))
targets = torch.from_numpy(np.hstack(v_data[:, 0]))

train_in, test_in, train_tar, test_tar  = train_test_split(inputs, targets, test_size=0.2, random_state=seed)

# Setup GP
gp_type = ZeroMeanAffineGP
likelihood = gpytorch.likelihoods.GaussianLikelihood()
gp = GaussianProcess(gp_type, likelihood, 1, output_dir)

#gp.train(train_in, train_tar.squeeze()) #, n_train=n_train, learning_rate=lr)
gp.init_with_hyperparam(output_dir)

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
plt.plot(cov_from_gamma-covs)

plt.show()
dummy = 0

