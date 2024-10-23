

import csv
import os
import shutil
import time
from copy import deepcopy
from datetime import datetime
from functools import partial

import casadi as cs
import gpytorch
import munch
import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver, AcadosSimSolver
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.model_selection import train_test_split
from skopt.sampler import Lhs
from termcolor import colored

from safe_control_gym.controllers.lqr.lqr_utils import discretize_linear_system
from safe_control_gym.controllers.mpc.gp_utils import (GaussianProcessCollection, ZeroMeanIndependentGPModel,
                                                       covSEard, kmeans_centriods, GaussianProcess)
from safe_control_gym.controllers.mpc.linear_mpc import MPC, LinearMPC
from safe_control_gym.controllers.mpc.mpc import MPC
from safe_control_gym.controllers.mpc.gpmpc_base import GPMPC
from safe_control_gym.controllers.mpc.mpc_acados import MPC_ACADOS
from safe_control_gym.envs.benchmark_env import Task
from safe_control_gym.utils.utils import timing

class GPMPC_ACADOS_TP(GPMPC):
    '''Implements a GP-MPC controller with Acados optimization.'''

    def __init__(
            self,
            env_func,
            seed: int = 1337,
            horizon: int = 5,
            q_mpc: list = [1],
            r_mpc: list = [1],
            constraint_tol: float = 1e-8,
            additional_constraints: list = None,
            soft_constraints: dict = None,
            warmstart: bool = True,
            train_iterations: int = None,
            test_data_ratio: float = 0.2,
            overwrite_saved_data: bool = True,
            optimization_iterations: list = None,
            learning_rate: list = None,
            normalize_training_data: bool = False,
            use_gpu: bool = False,
            gp_model_path: str = None,
            n_ind_points: int = 30,
            inducing_point_selection_method='kmeans',
            recalc_inducing_points_at_every_step=False,
            prob: float = 0.955,
            initial_rollout_std: float = 0.005,
            input_mask: list = None,
            target_mask: list = None,
            gp_approx: str = 'mean_eq',
            online_learning: bool = False,
            prior_info: dict = None,
            sparse_gp: bool = False,
            # inertial_prop: list = [1.0],
            prior_param_coeff: float = 1.0,
            terminate_run_on_done: bool = True,
            output_dir: str = 'results/temp',
            compute_ipopt_initial_guess: bool = True,
            use_RTI: bool = False,
            use_linear_prior: bool = True,
            **kwargs
    ):
        super().__init__(
            env_func=env_func,
            seed=seed,
            horizon=horizon,
            q_mpc=q_mpc,
            r_mpc=r_mpc,
            constraint_tol=constraint_tol,
            additional_constraints=additional_constraints,
            soft_constraints=soft_constraints,
            warmstart=warmstart,
            train_iterations=train_iterations,
            test_data_ratio=test_data_ratio,
            overwrite_saved_data=overwrite_saved_data,
            optimization_iterations=optimization_iterations,
            learning_rate=learning_rate,
            normalize_training_data=normalize_training_data,
            use_gpu=use_gpu,
            gp_model_path=gp_model_path,
            prob=prob,
            initial_rollout_std=initial_rollout_std,
            input_mask=input_mask,
            target_mask=target_mask,
            gp_approx=gp_approx,
            sparse_gp=sparse_gp,
            n_ind_points=n_ind_points,
            inducing_point_selection_method=inducing_point_selection_method,
            recalc_inducing_points_at_every_step=recalc_inducing_points_at_every_step,
            online_learning=online_learning,
            prior_info=prior_info,
            prior_param_coeff=prior_param_coeff,
            terminate_run_on_done=terminate_run_on_done,
            output_dir=output_dir,
            **kwargs)
        self.uncertain_dim = [1, 3, 5]
        self.Bd = np.eye(self.model.nx)[:, self.uncertain_dim]
        self.input_mask = None
        self.target_mask = None

        # MPC params
        # self.use_linear_prior = use_linear_prior
        self.use_linear_prior = False
        self.init_solver = 'ipopt'
        self.compute_ipopt_initial_guess = False
        self.use_RTI = use_RTI

        if hasattr(self, 'prior_ctrl'):
            self.prior_ctrl.close()
            
        if self.use_linear_prior:
            self.prior_ctrl = LinearMPC(
                self.prior_env_func,
                horizon=horizon,
                q_mpc=q_mpc,
                r_mpc=r_mpc,
                warmstart=warmstart,
                soft_constraints=self.soft_constraints_params['prior_soft_constraints'],
                terminate_run_on_done=terminate_run_on_done,
                prior_info=prior_info,
                # runner args
                # shared/base args
                output_dir=output_dir,
                additional_constraints=additional_constraints,
            )
        else:
            self.prior_ctrl = MPC_ACADOS(
                env_func=self.prior_env_func,
                horizon=horizon,
                q_mpc=q_mpc,
                r_mpc=r_mpc,
                warmstart=warmstart,
                soft_constraints=self.soft_constraints_params['prior_soft_constraints'],
                terminate_run_on_done=terminate_run_on_done,
                constraint_tol=constraint_tol,
                output_dir=output_dir,
                additional_constraints=additional_constraints,
                use_gpu=use_gpu,
                seed=seed,
                use_RTI=use_RTI,
                prior_info=prior_info,
            )
        self.prior_ctrl.reset()
        print('prior_ctrl:', type(self.prior_ctrl))
        if self.use_linear_prior:
            self.prior_dynamics_func = self.prior_ctrl.linear_dynamics_func
        else:
            self.prior_dynamics_func = self.prior_ctrl.dynamics_func
            self.prior_dynamcis_func_c = self.prior_ctrl.model.fc_func

        self.x_guess = None
        self.u_guess = None
        self.x_prev = None
        self.u_prev = None
        print('prior_info[prior_prop]', prior_info['prior_prop'])


    def preprocess_training_data(self,
                                 x_seq,
                                 u_seq,
                                 x_next_seq
                                 ):
        '''Converts trajectory data for GP trianing.

        Args:
            x_seq (list): state sequence of np.array (nx,).
            u_seq (list): action sequence of np.array (nu,).
            x_next_seq (list): next state sequence of np.array (nx,).

        Returns:
            np.array: inputs for GP training, (N, nx+nu).
            np.array: targets for GP training, (N, nx).
        '''
        # Get the predicted dynamics. This is a linear prior, thus we need to account for the fact that
        # it is linearized about an eq using self.X_GOAL and self.U_GOAL.
        g = 9.81
        dt = 1/60
        # x_pred_seq = self.prior_dynamics_func(x0=x_seq.T, p=u_seq.T)['xf'].toarray()
        acc = u_seq[:, 0]
        T_prior = self.prior_ctrl.env.T_mapping_func(acc).full().flatten()

        x_dot_seq = [(x_next_seq[i, :] - x_seq[i, :])/dt for i in range(x_seq.shape[0])]
        x_dot_seq = np.array(x_dot_seq)
        T_true = np.sqrt((x_dot_seq[:, 3] +  g) ** 2 + (x_dot_seq[:, 1] ** 2))
        
        targets_T = (T_true - T_prior).reshape(-1, 1)
        input_T = u_seq[:, 0].reshape(-1, 1)

        theta_true = x_dot_seq[:, 5]
        theta_prior = self.prior_dynamics_func(x0=x_seq.T, p=u_seq.T)['xf'].toarray()[5]
        targets_theta = (theta_true - theta_prior).reshape(-1, 1)
        input_theta = np.concatenate([x_seq[:, 4].reshape(-1, 1), 
                                      x_seq[:, 5].reshape(-1, 1),
                                      u_seq[:, 1].reshape(-1, 1)], axis=1) 
        train_input = np.concatenate([input_T, input_theta], axis=1)
        train_output = np.concatenate([targets_T, targets_theta], axis=1)
        return train_input, train_output

    def learn(self, env=None):
        '''Performs multiple epochs learning.
        '''

        train_runs = {0: {}}
        test_runs = {0: {}}

        if self.same_train_initial_state:
            train_envs = []
            for epoch in range(self.num_epochs):
                train_envs.append(self.env_func(randomized_init=True, seed=self.seed))
                train_envs[epoch].action_space.seed(self.seed)
        else:
            train_env = self.env_func(randomized_init=True, seed=self.seed)
            train_env.action_space.seed(self.seed)
            train_envs = [train_env] * self.num_epochs
        # init_test_states = get_random_init_states(env_func, num_test_episodes_per_epoch)
        test_envs = []
        if self.same_test_initial_state:
            for epoch in range(self.num_epochs):
                test_envs.append(self.env_func(randomized_init=True, seed=self.seed))
                test_envs[epoch].action_space.seed(self.seed)
        else:
            test_env = self.env_func(randomized_init=True, seed=self.seed)
            test_env.action_space.seed(self.seed)
            test_envs = [test_env] * self.num_epochs

        for episode in range(self.num_train_episodes_per_epoch):
            run_results = self.prior_ctrl.run(env=train_envs[0],
                                              terminate_run_on_done=self.terminate_train_on_done)
            train_runs[0].update({episode: munch.munchify(run_results)})
            self.reset()
        for test_ep in range(self.num_test_episodes_per_epoch):
            run_results = self.run(env=test_envs[0],
                                   terminate_run_on_done=self.terminate_test_on_done)
            test_runs[0].update({test_ep: munch.munchify(run_results)})
        self.reset()

        training_results = None
        for epoch in range(1, self.num_epochs):
            # only take data from the last episode from the last epoch
            if self.rand_data_selection:
                x_seq, actions, x_next_seq, x_dot_seq = self.gather_training_samples(train_runs, epoch - 1, self.num_samples, train_envs[epoch - 1].np_random)
            else:
                x_seq, actions, x_next_seq, x_dot_seq = self.gather_training_samples(train_runs, epoch - 1, self.num_samples)
            train_inputs, train_outputs = self.preprocess_training_data(x_seq, actions, x_next_seq)
            training_results = self.train_gp(input_data=train_inputs, target_data=train_outputs)
            # plot training results
            # if self.plot_trained_gp:
            #     self.gaussian_process.plot_trained_gp(train_inputs, train_outputs,
            #                                           output_dir=self.output_dir,
            #                                           title=f'epoch_{epoch}',
            #                                           residual_func=self.residual_func
            #                                           )
                
            # max_steps = train_runs[epoch-1][0]['obs'].shape[0]
            # x_seq, actions, x_next_seq, x_dot_seq = self.gather_training_samples(train_runs, epoch - 1, max_steps)
            # input_T, input_theta, targets_T, targets_theta = self.preprocess_training_data(x_seq, actions, x_next_seq)
            # # if self.plot_trained_gp:
            # #     self.gaussian_process.plot_trained_gp(test_inputs, test_outputs,
            # #                                           output_dir=self.output_dir,
            # #                                           title=f'epoch_{epoch}_train',
            # #                                           residual_func=self.residual_func
            # #                                           )
                
            # Test new policy.
            test_runs[epoch] = {}
            for test_ep in range(self.num_test_episodes_per_epoch):
                self.x_prev = test_runs[epoch - 1][episode]['obs'][:self.T + 1, :].T
                self.u_prev = test_runs[epoch - 1][episode]['action'][:self.T, :].T
                self.reset()
                run_results = self.run(env=test_envs[epoch],
                                       terminate_run_on_done=self.terminate_test_on_done)
                test_runs[epoch].update({test_ep: munch.munchify(run_results)})
            # max_steps = test_runs[epoch][0]['obs'].shape[0]
            # x_seq, actions, x_next_seq, x_dot_seq = self.gather_training_samples(test_runs, epoch - 1, max_steps)
            # input_T, input_theta, targets_T, targets_theta = self.preprocess_training_data(x_seq, actions, x_next_seq)
            # # if self.plot_trained_gp:
            # #     self.gaussian_process.plot_trained_gp(test_inputs, test_outputs,
            # #                                           output_dir=self.output_dir,
            # #                                           title=f'epoch_{epoch}_test',
            # #                                           residual_func=self.residual_func
            # #                                           )

            # gather training data
            train_runs[epoch] = {}
            for episode in range(self.num_train_episodes_per_epoch):
                self.reset()
                self.x_prev = train_runs[epoch - 1][episode]['obs'][:self.T + 1, :].T
                self.u_prev = train_runs[epoch - 1][episode]['action'][:self.T, :].T
                run_results = self.run(env=train_envs[epoch],
                                       terminate_run_on_done=self.terminate_train_on_done)
                train_runs[epoch].update({episode: munch.munchify(run_results)})

            # lengthscale, outputscale, noise, kern = self.gaussian_process.get_hyperparameters(as_numpy=True)
            # compute the condition number of the kernel matrix
            # TODO: fix data logging
            np.savez(os.path.join(self.output_dir, 'data_%s'% epoch),
                    data_inputs=training_results['train_inputs'],
                    data_targets=training_results['train_targets'],
                    train_runs=train_runs,
                    test_runs=test_runs,
                    num_epochs=self.num_epochs,
                    num_train_episodes_per_epoch=self.num_train_episodes_per_epoch,
                    num_test_episodes_per_epoch=self.num_test_episodes_per_epoch,
                    num_samples=self.num_samples,
                    # trajectory=self.trajectory,
                    # ctrl_freq=self.config.task_config.ctrl_freq,
                    # lengthscales=lengthscale,
                    # outputscale=outputscale,
                    # noise=noise,
                    # kern=kern,
                    train_data=self.train_data,
                    test_data=self.test_data,
                    )

        if training_results:
            np.savez(os.path.join(self.output_dir, 'data'),
                    data_inputs=training_results['train_inputs'],
                    data_targets=training_results['train_targets'])

        # close environments
        for env in train_envs:
            env.close()
        for env in test_envs:
            env.close()

        self.train_runs = train_runs
        self.test_runs = test_runs

        return train_runs, test_runs
    
    @timing
    def train_gp(self,
                 input_data, target_data,
                 gp_model=None,
                 overwrite_saved_data: bool = None,
                 train_hardware_data: bool = False,
                 ):
        '''Performs GP training.

        Args:
            input_data, target_data (optiona, np.array): data to use for training
            gp_model (str): if not None, this is the path to pretrained models to use instead of training new ones.
            overwrite_saved_data (bool): Overwrite the input and target data to the already saved data if it exists.
            train_hardware_data (bool): True to train on hardware data. If true, will load the data and perform training.
        Returns:
            training_results (dict): Dictionary of the training results.
        '''
        if gp_model is None and not train_hardware_data:
            gp_model = self.gp_model_path
        if overwrite_saved_data is None:
            overwrite_saved_data = self.overwrite_saved_data
        self.reset()
        train_inputs = input_data
        train_targets = target_data
        if (self.data_inputs is None and self.data_targets is None) or overwrite_saved_data:
            self.data_inputs = train_inputs
            self.data_targets = train_targets
        else:
            self.data_inputs = np.vstack((self.data_inputs, train_inputs))
            self.data_targets = np.vstack((self.data_targets, train_targets))

        total_input_data = self.data_inputs.shape[0]
        # If validation set is desired.
        if self.test_data_ratio > 0 and self.test_data_ratio is not None:
            train_idx, test_idx = train_test_split(
                list(range(total_input_data)),
                test_size=self.test_data_ratio,
                random_state=self.seed
            )

        else:
            # Otherwise, just copy the training data into the test data.
            train_idx = list(range(total_input_data))
            test_idx = list(range(total_input_data))

        train_inputs = self.data_inputs[train_idx, :]
        train_targets = self.data_targets[train_idx, :]
        self.train_data = {'train_inputs': train_inputs, 'train_targets': train_targets}
        test_inputs = self.data_inputs[test_idx, :]
        test_targets = self.data_targets[test_idx, :]
        self.test_data = {'test_inputs': test_inputs, 'test_targets': test_targets}


        train_inputs_tensor = torch.Tensor(train_inputs).double()
        train_targets_tensor = torch.Tensor(train_targets).double()
        test_inputs_tensor = torch.Tensor(test_inputs).double()
        test_targets_tensor = torch.Tensor(test_targets).double()

        # seperate the data for T and P
        train_input_T = train_inputs_tensor[:, 0].reshape(-1)
        train_target_T = train_targets_tensor[:, 0].reshape(-1)
        test_inputs_T = test_inputs_tensor[:, 0].reshape(-1)
        test_targets_T = test_targets_tensor[:, 0].reshape(-1)
        train_input_P = train_inputs_tensor[:, 1:].reshape(-1, 3)
        test_inputs_P = test_inputs_tensor[:, 1:].reshape(-1, 3)
        train_target_P = train_targets_tensor[:, 1:].reshape(-1)
        test_targets_P = test_targets_tensor[:, 1:].reshape(-1)

        # Define likelihood.
        likelihood_T = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(1e-6),
        ).double()
        likelihood_P = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(1e-6),
        ).double()

        GP_T = GaussianProcess(
            model_type=ZeroMeanIndependentGPModel,
            likelihood=likelihood_T,
            kernel='RBF_single', 
        )

        GP_P = GaussianProcess(
            model_type=ZeroMeanIndependentGPModel,
            likelihood=likelihood_P,
            kernel='RBF_single',
        )

        GP_T.train(train_input_T, train_target_T, test_inputs_T, test_targets_T,
                 n_train=self.optimization_iterations[0], learning_rate=self.learning_rate[0], 
                 gpu=self.use_gpu, fname='best_model_T')
        GP_P.train(train_input_P, train_target_P, test_inputs_P, test_targets_P,
                 n_train=self.optimization_iterations[1], learning_rate=self.learning_rate[1],
                 gpu=self.use_gpu, fname='best_model_P')

        self.gaussian_process = [GP_T, GP_P]
        

        # self.gaussian_process = GaussianProcessCollection(ZeroMeanIndependentGPModel,
        #                                                   likelihood,
        #                                                   len(self.target_mask),
        #                                                   input_mask=self.input_mask,
        #                                                   target_mask=self.target_mask,
        #                                                   normalize=self.normalize_training_data,
        #                                                   kernel=self.kernel,
        #                                                   parallel=self.parallel
        #                                                   )
        # if gp_model:
        #     self.gaussian_process.init_with_hyperparam(train_inputs_tensor,
        #                                                train_targets_tensor,
        #                                                gp_model)
        #     print(colored(f'Loaded pretrained model from {gp_model}', 'green'))
        # else:
            # Train the GP.
        # self.gaussian_process.train(train_inputs_tensor,
        #                             train_targets_tensor,
        #                             test_inputs_tensor,
        #                             test_targets_tensor,
        #                             n_train=self.optimization_iterations,
        #                             learning_rate=self.learning_rate,
        #                             gpu=self.use_gpu,
        #                             output_dir=self.output_dir)
        

        self.reset()
        # if self.train_data['train_targets'].shape[0] <= self.n_ind_points:
        #    n_ind_points = self.train_data['train_targets'].shape[0]
        # else:
        #    n_ind_points = self.n_ind_points
        # self.set_gp_dynamics_func(n_ind_points)
        # self.setup_gp_optimizer(n_ind_points)
        # Collect training results.
        training_results = {}
        training_results['train_targets'] = train_targets
        training_results['train_inputs'] = train_inputs
        return training_results

    def setup_acados_model(self, n_ind_points) -> AcadosModel:

        # setup GP related
        self.inverse_cdf = scipy.stats.norm.ppf(1 - (1 / self.model.nx - (self.prob + 1) / (2 * self.model.nx)))
        if self.sparse_gp:
            self.create_sparse_GP_machinery(n_ind_points)

        # setup acados model
        model_name = self.env.NAME

        acados_model = AcadosModel()
        acados_model.x = self.model.x_sym
        acados_model.u = self.model.u_sym
        acados_model.name = model_name

        A_lin = self.discrete_dfdx
        B_lin = self.discrete_dfdu

        z = cs.vertcat(acados_model.x, acados_model.u)  # GP prediction point
        # z = z[self.input_mask]

        # full_dyn = self.prior_dynamics_func(x0=acados_model.x- cs.MX(self.prior_ctrl.X_EQ[:, None]),
        #                                     p=acados_model.u- cs.MX(self.prior_ctrl.U_EQ[:, None]))['xf'] \
        #     + self.prior_ctrl.X_EQ[:, None] \
        #     + self.Bd @ self.gaussian_process.casadi_predict(z=z)['mean']
        # self.full_func = cs.Function('full_func', [acados_model.x, acados_model.u], [full_dyn])

        if self.sparse_gp:
            # sparse GP inducing points
            '''
            z_ind should be of shape (n_ind_points, z.shape[0]) or (n_ind_points, len(self.input_mask))
            mean_post_factor should be of shape (len(self.target_mask), n_ind_points)
            Here we create the corresponding parameters since acados supports only 1D parameters
            '''
            z_ind = cs.MX.sym('z_ind', n_ind_points, len(self.input_mask))
            mean_post_factor = cs.MX.sym('mean_post_factor', len(self.target_mask), n_ind_points)
            acados_model.p = cs.vertcat(cs.reshape(z_ind, -1, 1), cs.reshape(mean_post_factor, -1, 1))
            # define the dynamics
            # f_disc = self.prior_dynamics_func(x0=acados_model.x, p=acados_model.u)['xf'] \
            #     + 

            self.sparse_gp_func = cs.Function('sparse_func',
                                              [acados_model.x, acados_model.u, z_ind, mean_post_factor], [f_disc])
        else:
            # f_disc = self.prior_dynamics_func(x0=acados_model.x, p=acados_model.u)['xf'] \
            #         + self.Bd @ self.gaussian_process.casadi_predict(z=z)['mean']
            GP_T = self.gaussian_process[0]
            GP_P = self.gaussian_process[1]
            T_pred_point = z[6]
            P_pred_point = z[[4, 5, 7]]
            casadi_pred_T = GP_T.casadi_predict(z=T_pred_point)['mean']
            casadi_pred_P = GP_P.casadi_predict(z=P_pred_point)['mean']
            f_cont = self.prior_dynamcis_func_c(x=acados_model.x, u=acados_model.u)['f']\
                    + cs.vertcat(0, cs.sin(acados_model.x[4])*casadi_pred_T, 
                                 0, cs.cos(acados_model.x[4])*casadi_pred_T,
                                 0, casadi_pred_P)
            
            f_cont_func = cs.Function('f_cont_func', [acados_model.x, acados_model.u], [f_cont])
            # use rk4 to discretize the continuous dynamics
            k1 = f_cont_func(acados_model.x, acados_model.u)
            k2 = f_cont_func(acados_model.x + self.dt/2 * k1, acados_model.u)
            k3 = f_cont_func(acados_model.x + self.dt/2 * k2, acados_model.u)
            k4 = f_cont_func(acados_model.x + self.dt * k3, acados_model.u)
            f_disc = acados_model.x + self.dt/6 * (k1 + 2*k2 + 2*k3 + k4)

        acados_model.disc_dyn_expr = f_disc

        acados_model.x_labels = self.env.STATE_LABELS
        acados_model.u_labels = self.env.ACTION_LABELS
        acados_model.t_label = 'time'

        self.acados_model = acados_model

    def setup_acados_optimizer(self, n_ind_points):
        print('=================Setting up acados optimizer=================')
        # before_optimizer_setup = time.time()
        nx, nu = self.model.nx, self.model.nu
        ny = nx + nu
        ny_e = nx

        # create ocp object to formulate the OCP
        ocp = AcadosOcp()
        ocp.model = self.acados_model

        # set dimensions
        ocp.dims.N = self.T  # prediction horizon

        # set cost
        ocp.cost.cost_type = 'LINEAR_LS'
        ocp.cost.cost_type_e = 'LINEAR_LS'
        # cost weight matrices
        ocp.cost.W = scipy.linalg.block_diag(self.Q, self.R)
        ocp.cost.W_e = self.P if hasattr(self, 'P') else self.Q
        # ocp.cost.W_e = self.Q

        ocp.cost.Vx = np.zeros((ny, nx))
        ocp.cost.Vx[:nx, :nx] = np.eye(nx)
        ocp.cost.Vu = np.zeros((ny, nu))
        ocp.cost.Vu[nx:(nx + nu), :nu] = np.eye(nu)
        ocp.cost.Vx_e = np.eye(nx)
        # placeholder y_ref and y_ref_e (will be set in select_action)
        ocp.cost.yref = np.zeros((ny, ))
        ocp.cost.yref_e = np.zeros((ny_e, ))

        # Constraints
        # general constraint expressions
        state_constraint_expr_list = []
        input_constraint_expr_list = []
        state_tighten_list = []
        input_tighten_list = []
        for sc_i, state_constraint in enumerate(self.state_constraints_sym):
            state_constraint_expr_list.append(state_constraint(ocp.model.x))
            # chance state constraint tightening
            state_tighten_list.append(cs.MX.sym(f'state_tighten_{sc_i}', state_constraint(ocp.model.x).shape[0], 1))
        for ic_i, input_constraint in enumerate(self.input_constraints_sym):
            input_constraint_expr_list.append(input_constraint(ocp.model.u))
            # chance input constraint tightening
            input_tighten_list.append(cs.MX.sym(f'input_tighten_{ic_i}', input_constraint(ocp.model.u).shape[0], 1))

        h_expr_list = state_constraint_expr_list + input_constraint_expr_list
        h_expr = cs.vertcat(*h_expr_list)
        h0_expr = cs.vertcat(*h_expr_list)
        he_expr = cs.vertcat(*state_constraint_expr_list)  # terminal constraints are only state constraints
        # pass the constraints to the ocp object
        ocp = self.processing_acados_constraints_expression(ocp, h0_expr, h_expr, he_expr, state_tighten_list, input_tighten_list)
        # pass the tightening variables to the ocp object as parameters
        tighten_param = cs.vertcat(*state_tighten_list, *input_tighten_list)
        if self.sparse_gp:
            ocp.model.p = cs.vertcat(ocp.model.p, tighten_param)
        else:
            ocp.model.p = tighten_param
        ocp.parameter_values = np.zeros((ocp.model.p.shape[0], ))  # dummy values

        # slack costs for nonlinear constraints
        if self.gp_soft_constraints:
            # slack variables for all constraints
            ocp.constraints.Jsh_0 = np.eye(h0_expr.shape[0])
            ocp.constraints.Jsh = np.eye(h_expr.shape[0])
            ocp.constraints.Jsh_e = np.eye(he_expr.shape[0])
            # slack penalty (TODO: using the value specified in the config)
            L2_pen = self.gp_soft_constraints_coeff
            L1_pen = self.gp_soft_constraints_coeff
            ocp.cost.zl_0 = L1_pen * np.ones(h0_expr.shape[0])
            ocp.cost.zu_0 = L1_pen * np.ones(h0_expr.shape[0])
            ocp.cost.Zu_0 = L2_pen * np.ones(h0_expr.shape[0])
            ocp.cost.Zl_0 = L2_pen * np.ones(h0_expr.shape[0])
            ocp.cost.Zu = L2_pen * np.ones(h_expr.shape[0])
            ocp.cost.Zl = L2_pen * np.ones(h_expr.shape[0])
            ocp.cost.zl = L1_pen * np.ones(h_expr.shape[0])
            ocp.cost.zu = L1_pen * np.ones(h_expr.shape[0])
            ocp.cost.Zl_e = L2_pen * np.ones(he_expr.shape[0])
            ocp.cost.Zu_e = L2_pen * np.ones(he_expr.shape[0])
            ocp.cost.zl_e = L1_pen * np.ones(he_expr.shape[0])
            ocp.cost.zu_e = L1_pen * np.ones(he_expr.shape[0])

        # placeholder initial state constraint
        x_init = np.zeros((nx))
        ocp.constraints.x0 = x_init

        # set up solver options
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        # ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.integrator_type = 'DISCRETE'
        ocp.solver_options.nlp_solver_type = 'SQP' if not self.use_RTI else 'SQP_RTI'
        ocp.solver_options.nlp_solver_max_iter = 10 if not self.use_RTI else 1
        ocp.solver_options.qp_solver_iter_max = 10
        ocp.solver_options.qp_tol = 1e-4
        ocp.solver_options.tol = 1e-4
        ocp.solver_options.as_rti_level = 0 if not self.use_RTI else 4
        ocp.solver_options.as_rti_iter = 1 if not self.use_RTI else 1

        # ocp.solver_options.globalization = 'FUNNEL_L1PEN_LINESEARCH' if not self.use_RTI else 'MERIT_BACKTRACKING'
        # ocp.solver_options.globalization = 'MERIT_BACKTRACKING'
        # prediction horizon
        ocp.solver_options.tf = self.T * self.dt

        # c code generation
        # NOTE: when using GP-MPC, a separated directory is needed;
        # otherwise, Acados solver can read the wrong c code
        ocp.code_export_directory = self.output_dir + '/gpmpc_c_generated_code'

        self.ocp = ocp
        self.opti_dict = {'n_ind_points': n_ind_points}
        # compute sparse GP values
        # the actual values will be set in select_action_with_gp
        # if self.sparse_gp:
        #     mean_post_factor_val, _, _, z_ind_val = self.precompute_sparse_gp_values(n_ind_points)
        #     self.mean_post_factor_val = mean_post_factor_val
        #     self.z_ind_val = z_ind_val
        # else:
        #     mean_post_factor_val, z_ind_val = self.precompute_mean_post_factor_all_data()
        #     self.mean_post_factor_val = mean_post_factor_val
        #     self.z_ind_val = z_ind_val

    def processing_acados_constraints_expression(self, ocp: AcadosOcp, h0_expr, h_expr, he_expr,
                                                 state_tighten_list, input_tighten_list) -> AcadosOcp:
        '''Preprocess the constraints to be compatible with acados.
            Args:
                h0_expr (casadi expression): initial state constraints
                h_expr (casadi expression): state and input constraints
                he_expr (casadi expression): terminal state constraints
                state_tighten_list (list): list of casadi SX variables for state constraint tightening
                input_tighten_list (list): list of casadi SX variables for input constraint tightening
            Returns:
                ocp (AcadosOcp): acados ocp object with constraints set

        Note:
        all constraints in safe-control-gym are defined as g(x, u) <= constraint_tol
        However, acados requires the constraints to be defined as lb <= g(x, u) <= ub
        Thus, a large negative number (-1e8) is used as the lower bound.
        See: https://github.com/acados/acados/issues/650

        An alternative way to set the constraints is to use bounded constraints of acados:
        # bounded input constraints
        idxbu = np.where(np.sum(self.env.constraints.input_constraints[0].constraint_filter, axis=0) != 0)[0]
        ocp.constraints.Jbu = np.eye(nu)
        ocp.constraints.lbu = self.env.constraints.input_constraints[0].lower_bounds
        ocp.constraints.ubu = self.env.constraints.input_constraints[0].upper_bounds
        ocp.constraints.idxbu = idxbu # active constraints dimension
        '''
        # NOTE: only the upper bound is tightened due to constraint are defined in the
        # form of g(x, u) <= constraint_tol in safe-control-gym

        # lambda functions to set the upper and lower bounds of the chance constraints
        def constraint_ub_chance(constraint): return -self.constraint_tol * np.ones(constraint.shape)
        def constraint_lb_chance(constraint): return -1e8 * np.ones(constraint.shape)
        state_tighten_var = cs.vertcat(*state_tighten_list)
        input_tighten_var = cs.vertcat(*input_tighten_list)

        ub = {'h': constraint_ub_chance(h_expr - cs.vertcat(state_tighten_var, input_tighten_var)),
              'h0': constraint_ub_chance(h0_expr - cs.vertcat(state_tighten_var, input_tighten_var)),
              'he': constraint_ub_chance(he_expr - state_tighten_var)}
        lb = {'h': constraint_lb_chance(h_expr),
              'h0': constraint_lb_chance(h0_expr),
              'he': constraint_lb_chance(he_expr)}

        # make sure all the ub and lb are 1D casaadi SX variables
        # (see: https://discourse.acados.org/t/infeasible-qps-when-using-nonlinear-casadi-constraint-expressions/1595/5?u=mxche)
        for key in ub.keys():
            ub[key] = ub[key].flatten() if ub[key].ndim != 1 else ub[key]
            lb[key] = lb[key].flatten() if lb[key].ndim != 1 else lb[key]
        # check ub and lb dimensions
        for key in ub.keys():
            assert ub[key].ndim == 1, f'ub[{key}] is not 1D numpy array'
            assert lb[key].ndim == 1, f'lb[{key}] is not 1D numpy array'
        assert ub['h'].shape == lb['h'].shape, 'h_ub and h_lb have different shapes'

        # pass the constraints to the ocp object
        ocp.model.con_h_expr_0 = h0_expr - cs.vertcat(state_tighten_var, input_tighten_var)
        ocp.model.con_h_expr = h_expr - cs.vertcat(state_tighten_var, input_tighten_var)
        ocp.model.con_h_expr_e = he_expr - state_tighten_var
        ocp.dims.nh_0, ocp.dims.nh, ocp.dims.nh_e = \
            h0_expr.shape[0], h_expr.shape[0], he_expr.shape[0]
        # assign constraints upper and lower bounds
        ocp.constraints.uh_0 = ub['h0']
        ocp.constraints.lh_0 = lb['h0']
        ocp.constraints.uh = ub['h']
        ocp.constraints.lh = lb['h']
        ocp.constraints.uh_e = ub['he']
        ocp.constraints.lh_e = lb['he']

        return ocp

    @timing
    def select_action(self, obs, info=None):
        time_before = time.time()
        if self.gaussian_process is None:
            action = self.prior_ctrl.select_action(obs)
        else:
            action = self.select_action_with_gp(obs)
        time_after = time.time()
        self.results_dict['runtime'].append(time_after - time_before)
        print('current obs:', obs)
        print('current action:', action)
        self.last_obs = obs
        self.last_action = action

        return action

    # @timing
    def select_action_with_gp(self, obs):
        time_before = time.time()
        nx, nu = self.model.nx, self.model.nu
        ny = nx + nu
        ny_e = nx
        # TODO: replace this with something safer
        n_ind_points = self.opti_dict['n_ind_points']

        # set initial condition (0-th state)
        self.acados_ocp_solver.set(0, 'lbx', obs)
        self.acados_ocp_solver.set(0, 'ubx', obs)
        # set initial guess for the solution
        if self.warmstart:
            if self.x_guess is None or self.u_guess is None:
                if self.compute_ipopt_initial_guess:
                    # compute initial guess with IPOPT
                    self.compute_initial_guess(obs, self.get_references())
                else:
                    # use zero initial guess (TODO: use acados warm start)
                    self.x_guess = np.zeros((nx, self.T + 1))
                    if nu == 1:
                        self.u_guess = np.zeros((self.T,))
                    else:
                        self.u_guess = np.zeros((nu, self.T))
            # set initial guess
            for idx in range(self.T + 1):
                init_x = self.x_guess[:, idx]
                self.acados_ocp_solver.set(idx, 'x', init_x)
            for idx in range(self.T):
                if nu == 1:
                    init_u = np.array([self.u_guess[idx]])
                else:
                    init_u = self.u_guess[:, idx]
                self.acados_ocp_solver.set(idx, 'u', init_u)
        else:
            for idx in range(self.T + 1):
                self.acados_ocp_solver.set(idx, 'x', obs)
            for idx in range(self.T):
                self.acados_ocp_solver.set(idx, 'u', np.zeros((nu,)))

        # compute the sparse GP values
        '''
        if self.recalc_inducing_points_at_every_step:
            mean_post_factor_val, _, _, z_ind_val = self.precompute_sparse_gp_values(n_ind_points)
            self.results_dict['inducing_points'].append(z_ind_val)
        else:
            # use the precomputed values
            mean_post_factor_val = self.mean_post_factor_val
            z_ind_val = self.z_ind_val
            self.results_dict['inducing_points'] = [z_ind_val]
        '''
        # Set the probabilistic state and input constraint set limits.
        # Tightening at the first step is possible if self.compute_initial_guess is used

        state_constraint_set_prev, input_constraint_set_prev = self.precompute_probabilistic_limits()

        # set acados parameters
        if self.sparse_gp:
            # sparse GP parameters
            assert z_ind_val.shape == (n_ind_points, len(self.input_mask))
            assert mean_post_factor_val.shape == (len(self.target_mask), n_ind_points)
            # casadi use column major order, while np uses row major order by default
            # Thus, Fortran order (column major) is used to reshape the arrays
            z_ind_val = z_ind_val.reshape(-1, 1, order='F')
            mean_post_factor_val = mean_post_factor_val.reshape(-1, 1, order='F')
            dyn_value = np.concatenate((z_ind_val, mean_post_factor_val)).reshape(-1)
            # tighten constraints
            for idx in range(self.T):
                # tighten initial and path constraints
                state_constraint_set = state_constraint_set_prev[0][:, idx]
                input_constraint_set = input_constraint_set_prev[0][:, idx]
                tighten_value = np.concatenate((state_constraint_set, input_constraint_set))
                # set the parameter values
                parameter_values = np.concatenate((dyn_value, tighten_value))
                # self.acados_ocp_solver.set(idx, "p", dyn_value)
                self.acados_ocp_solver.set(idx, 'p', parameter_values)
            # tighten terminal state constraints
            tighten_value = np.concatenate((state_constraint_set_prev[0][:, self.T], np.zeros((2 * nu,))))
            # set the parameter values
            parameter_values = np.concatenate((dyn_value, tighten_value))
            self.acados_ocp_solver.set(self.T, 'p', parameter_values)
            # self.acados_ocp_solver.set(self.T, "p", dyn_value)
        else:
            for idx in range(self.T):
                # tighten initial and path constraints
                state_constraint_set = state_constraint_set_prev[0][:, idx]
                input_constraint_set = input_constraint_set_prev[0][:, idx]
                tighten_value = np.concatenate((state_constraint_set, input_constraint_set))
                self.acados_ocp_solver.set(idx, 'p', tighten_value)
            # tighten terminal state constraints
            tighten_value = np.concatenate((state_constraint_set_prev[0][:, self.T], np.zeros((2 * nu,))))
            self.acados_ocp_solver.set(self.T, 'p', tighten_value)


        # set reference for the control horizon
        goal_states = self.get_references()
        if self.mode == 'tracking':
            self.traj_step += 1
        for idx in range(self.T):
            y_ref = np.concatenate((goal_states[:, idx], np.zeros((nu,))))
            self.acados_ocp_solver.set(idx, 'yref', y_ref)
        y_ref_e = goal_states[:, -1]
        self.acados_ocp_solver.set(self.T, 'yref', y_ref_e)

        # solve the optimization problem
        if self.use_RTI:
            # preparation phase
            self.acados_ocp_solver.options_set('rti_phase', 1)
            status = self.acados_ocp_solver.solve()

            # feedback phase
            self.acados_ocp_solver.options_set('rti_phase', 2)
            status = self.acados_ocp_solver.solve()
        else:
            status = self.acados_ocp_solver.solve()
        if status not in [0, 2]:
            self.acados_ocp_solver.print_statistics()
            print(colored(f'acados returned status {status}. ', 'red'))

        action = self.acados_ocp_solver.get(0, 'u')
        # get the open-loop solution
        if self.x_prev is None and self.u_prev is None:
            self.x_prev = np.zeros((nx, self.T + 1))
            self.u_prev = np.zeros((nu, self.T))
        if self.u_prev is not None and nu == 1:
            self.u_prev = self.u_prev.reshape((1, -1))

        for i in range(self.T + 1):
            self.x_prev[:, i] = self.acados_ocp_solver.get(i, 'x')
        for i in range(self.T):
            self.u_prev[:, i] = self.acados_ocp_solver.get(i, 'u')
        if nu == 1:
            self.u_prev = self.u_prev.flatten()
        self.x_guess = self.x_prev
        self.u_guess = self.u_prev

        time_after = time.time()
        print(f'gpmpc acados sol time: {time_after - time_before:.3f}; sol status {status}; nlp iter {self.acados_ocp_solver.get_stats("sqp_iter")}; qp iter {self.acados_ocp_solver.get_stats("qp_iter")}')
        if time_after - time_before > 1 / 60:
            print(colored(f'========= Warning: GPMPC ACADOS took {time_after - time_before:.3f} seconds =========', 'yellow'))

        if hasattr(self, 'K'):
            action += self.K @ (self.x_prev[:, 0] - obs)
            # self.u_prev = self.u_prev + self.K @ (self.x_prev - obs)
            # self.u_guess = self.u_prev
            # action = self.u_prev[0] if nu == 1 else self.u_prev[:, 0]

        return action

    @timing
    def reset(self):
        '''Reset the controller before running.'''
        # Setup reference input.
        if self.env.TASK == Task.STABILIZATION:
            self.mode = 'stabilization'
            self.x_goal = self.env.X_GOAL
        elif self.env.TASK == Task.TRAJ_TRACKING:
            self.mode = 'tracking'
            self.traj = self.env.X_GOAL.T
            self.traj_step = 0
        # Dynamics model.
        self.setup_prior_dynamics()
        if self.gaussian_process is not None:
            # self.compute_terminal_cost_and_ancillary_gain()
            # sparse GP
            if self.sparse_gp and self.train_data['train_targets'].shape[0] <= self.n_ind_points:
                n_ind_points = self.train_data['train_targets'].shape[0]
            elif self.sparse_gp:
                n_ind_points = self.n_ind_points
            else:
                n_ind_points = self.train_data['train_targets'].shape[0]

            # explicitly clear the previously generated c code, ocp and solver
            # otherwise the number of parameters will be incorrect
            # TODO: find a better way to handle this
            self.acados_model = None
            self.ocp = None
            self.acados_ocp_solver = None
            # delete the generated c code directory
            if os.path.exists(self.output_dir + '/gpmpc_c_generated_code'):
                print('deleting the generated c code directory')
                shutil.rmtree(self.output_dir + '/gpmpc_c_generated_code', ignore_errors=False)
                assert not os.path.exists(self.output_dir + '/gpmpc_c_generated_code')

            # reinitialize the acados model and solver
            self.setup_acados_model(n_ind_points)
            self.setup_acados_optimizer(n_ind_points)
            self.acados_ocp_solver = AcadosOcpSolver(self.ocp, self.output_dir + '/gpmpc_acados_ocp_solver.json')

        self.prior_ctrl.reset()
        self.setup_results_dict()
        # Previously solved states & inputs, useful for warm start.
        self.x_prev = None
        self.u_prev = None

        self.x_guess = None
        self.u_guess = None
