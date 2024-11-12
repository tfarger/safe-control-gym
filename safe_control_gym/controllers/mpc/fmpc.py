'''Flatness based Model Predictive Control.

Based on:
    Linear MPC 
'''
#TODO s
# get Q and R matrices from config files
# reenable state constraints in linearMPC
# fix X_GOAL


from copy import deepcopy
from sys import platform

import casadi as cs
import numpy as np
import pickle

import matplotlib.pyplot as plt

from safe_control_gym.controllers.lqr.lqr_utils import discretize_linear_system, compute_lqr_gain
from safe_control_gym.controllers.mpc.mpc import MPC
from safe_control_gym.controllers.mpc.linear_mpc import LinearMPC
from safe_control_gym.controllers.mpc.mpc_utils import compute_discrete_lqr_gain_from_cont_linear_system, get_cost_weight_matrix
from safe_control_gym.envs.benchmark_env import Task

from safe_control_gym.math_and_models.symbolic_systems import SymbolicModel

from plottingUtils import *

class FlatMPC(LinearMPC):
    '''Flatness based MPC.'''

    def __init__(
            self,
            env_func,
            horizon=5,
            q_mpc=[1],
            r_mpc=[1],
            warmstart=True,
            soft_constraints=False,
            terminate_run_on_done=True,
            constraint_tol: float = 1e-8,
            solver: str = 'sqpmethod',
            # runner args
            # shared/base args
            output_dir='results/temp',
            additional_constraints=None,
            **kwargs):
        '''Creates task and controller.

        Args:
            env_func (Callable): function to instantiate task/environment.
            horizon (int): mpc planning horizon.
            q_mpc (list): diagonals of state cost weight.
            r_mpc (list): diagonals of input/action cost weight.
            warmstart (bool): if to initialize from previous iteration.
            soft_constraints (bool): Formulate the constraints as soft constraints.
            terminate_run_on_done (bool): Terminate the run when the environment returns done or not.
            constraint_tol (float): Tolerance to add the the constraint as sometimes solvers are not exact.
            solver (str): Specify which solver you wish to use (qrqp, qpoases, ipopt, sqpmethod)
            output_dir (str): output directory to write logs and results.
            additional_constraints (list): list of constraints.
        '''
        # Store all params/args.
        for k, v in locals().items():
            if k != 'self' and k != 'kwargs' and '__' not in k:
                self.__dict__[k] = v

        super().__init__(
            env_func,
            horizon=horizon,
            q_mpc=q_mpc,
            r_mpc=r_mpc,
            warmstart=warmstart,
            soft_constraints=soft_constraints,
            terminate_run_on_done=terminate_run_on_done,
            constraint_tol=constraint_tol,
            # prior_info=prior_info,
            output_dir=output_dir,
            additional_constraints=additional_constraints,
            **kwargs
        )

        # replace dynamics model with symbolic flat model
        self.model = self._setup_flat_model_symbolic()

        # TODO: setup environment equilibrium
        # self.X_EQ = np.atleast_2d(self.env.X_GOAL)[0,:].T
        # self.U_EQ = np.atleast_2d(self.env.U_GOAL)[0,:]

        self.X_EQ = np.atleast_2d(self.model.X_EQ)[0, :].T
        self.U_EQ = np.atleast_2d(self.model.U_EQ)[0, :].T 
        assert solver in ['qpoases', 'qrqp', 'sqpmethod', 'ipopt'], '[Error]. MPC Solver not supported.'
        self.solver = solver

        # overwrite definitions in parent init function to fit flat model
        self.Q = get_cost_weight_matrix([5, 0.1, 0.1, 0.1, 5, 0.1, 0.1, 0.1], self.model.nx) 
        self.R = get_cost_weight_matrix([0.00001], self.model.nu)

        self.u0_dot = -0.03174012 
        self.u0_prev = 0.33837254
        
        # self.x_prev_fmpc = np.array([0.76046276,  0.01807867,  0.98482524,  0.78595726, -0.08439172, -0.01873777])

        # temporarily use a LQR for testing        
        # self.gain = compute_lqr_gain(self.model, self.model.X_EQ, self.model.U_EQ,
        #                              self.Q, self.R, True)
        
        self.counter =0
        


    def _setup_flat_model_symbolic(self):
        '''Creates symbolic (CasADi) models for dynamics, observation, and cost.

        Args:
            prior_prop (dict): specify the prior inertial prop to use in the symbolic model.
        ''' 
        nx, nu = 8, 2
      
        dt = self.dt
        # Define states.
        z = cs.MX.sym('z')
        z_dot = cs.MX.sym('z_dot')
        z_ddot = cs.MX.sym('z_ddot')   
        z_dddot = cs.MX.sym('z_dddot')       
        
        x = cs.MX.sym('x')
        x_dot = cs.MX.sym('x_dot')
        x_ddot = cs.MX.sym('x_ddot')
        x_dddot = cs.MX.sym('x_dddot')
        
        X = cs.vertcat(x, x_dot, x_ddot, x_dddot, z, z_dot, z_ddot, z_dddot)
        # Define flat inputs 
        v1 = cs.MX.sym('v1')
        v2 = cs.MX.sym('v2')
        U = cs.vertcat(v1, v2)
        # Define dynamics equations.
        X_dot = cs.vertcat(x_dot, x_ddot, x_dddot, v1,
                           z_dot, z_ddot, z_dddot, v2)
        # Define observation.
        Y = cs.vertcat(x, x_dot, x_ddot, x_dddot, z, z_dot, z_ddot, z_dddot)
        
        # Set the equilibrium values for linearizations.
        X_EQ = np.zeros(nx)
        U_EQ = np.zeros(nu)
        # Define cost (quadratic form).
        Q = cs.MX.sym('Q', nx, nx)
        R = cs.MX.sym('R', nu, nu)
        Xr = cs.MX.sym('Xr', nx, 1)
        Ur = cs.MX.sym('Ur', nu, 1)
        cost_func = 0.5 * (X - Xr).T @ Q @ (X - Xr) + 0.5 * (U - Ur).T @ R @ (U - Ur)
        # Define dynamics and cost dictionaries.
        dynamics = {'dyn_eqn': X_dot, 'obs_eqn': Y, 'vars': {'X': X, 'U': U}}
        cost = {
            'cost_func': cost_func,
            'vars': {
                'X': X,
                'U': U,
                'Xr': Xr,
                'Ur': Ur,
                'Q': Q,
                'R': R
            }
        }
        # Additional params to cache
        params = {
            # equilibrium point for linearization
            'X_EQ': X_EQ,
            'U_EQ': U_EQ,
        }
        # Setup symbolic model.
        #self.symbolic = SymbolicModel(dynamics=dynamics, cost=cost, dt=dt, params=params)
        return SymbolicModel(dynamics=dynamics, cost=cost, dt=dt, params=params)


    # overwrite to input flat trajectory into reference
    def reset(self):
        '''Prepares for training or evaluation.'''
        # Setup reference input.
        if self.env.TASK == Task.STABILIZATION:
            self.mode = 'stabilization'
            self.x_goal = self.env.X_GOAL
        elif self.env.TASK == Task.TRAJ_TRACKING:
            self.mode = 'tracking'
            #self.traj = self.env.X_GOAL.T
            self.traj = _load_flat_trajectory_circle()
            # Step along the reference.
            self.traj_step = 0
        # Dynamics model.
        self.set_dynamics_func()
        # CasADi optimizer.
        self.setup_optimizer()
        # Previously solved states & inputs, useful for warm start.
        self.x_prev = None
        self.u_prev = None

        self.setup_results_dict()

    def setup_results_dict(self):
        '''Setup the results dictionary to store run information.'''
        self.results_dict = {'obs': [],
                             'reward': [],
                             'done': [],
                             'info': [],
                             'action': [],
                             'horizon_inputs': [],
                             'horizon_states': [],
                             'goal_states': [],
                             'frames': [],
                             'state_mse': [],
                             'common_cost': [],
                             'state': [],
                             'state_error': [],
                             't_wall': [],
                             # addition for flat MPC
                             'obs_x': [],
                             'obs_z': [],
                             'v': [],
                             'u': [],
                             'T_dot': [],
                             'u_ref': [],
                             'horizon_v': [],
                             'horizon_z': [],
                             'horizon_u': [],
                             }


    def select_action(self,
                      obs,
                      info=None
                      ):
        '''Solve nonlinear mpc problem to get next action.

        Args:
            obs (ndarray): Current state/observation.
            info (dict): Current info.

        Returns:
            action (ndarray): Input/action to the task/env.
        '''
        # NMPC input reference
        with open('/home/tobias/Studium/masterarbeit/code/safe-control-gym/examples/mpc/temp-data/reference_NMPC.pkl', 'rb') as file:
            data_dict = pickle.load(file)
        u_nmpc = data_dict['u_ref'].transpose()
        v_nmpc = data_dict['v_ref'].transpose()
        z_nmpc = data_dict['z_ref'].transpose()
        u_dot_nmpc = data_dict['u_dot_ref'].transpose()
        
        start = min(self.traj_step, self.traj.shape[-1])
        end = min(self.traj_step + self.T , self.traj.shape[-1])
        remain = max(0, self.T - (end - start))
        u_nmpc_horizon = np.concatenate([
            u_nmpc[:, start:end],
            np.tile(u_nmpc[:, -1:], (1, remain))
        ], -1)   
        v_nmpc_horizon = np.concatenate([
            v_nmpc[:, start:end],
            np.tile(v_nmpc[:, -1:], (1, remain))
        ], -1)  
        z_nmpc_horizon = np.concatenate([
            z_nmpc[:, start:end],
            np.tile(z_nmpc[:, -1:], (1, remain))
        ], -1)  
        u_dot_nmpc_horizon = np.concatenate([
            u_dot_nmpc[:, start:end],
            np.tile(u_dot_nmpc[:, -1:], (1, remain))
        ], -1) 


        nz, nv = self.model.nx, self.model.nu 
        # determine flat state from observation      
        u_ref_nmpc_prev = u_nmpc[:, start-1] # careful, this is wrong at time step 0
        # transform real states to flat states
        z_obs = _get_z_from_regular_states(obs, self.u0_prev, self.u0_dot) # does not work
        # z_obs = _get_z_from_regular_states(obs, u_nmpc_horizon[:,0], u_dot_nmpc_horizon[0, 0]+u_dot_nmpc_horizon[1,0]) 
        # z_obs = _get_z_from_regular_states(obs, self.u_prev_tmp, u_dot_nmpc_horizon[0, 0]+u_dot_nmpc_horizon[1,0])  
        # z_obs = _get_z_from_regular_states(obs, u_ref_nmpc_prev, u_dot_nmpc_horizon[0, 0]+u_dot_nmpc_horizon[1,0])  
        # z_obs = _get_z_from_regular_states_FD(obs, self.x_prev_fmpc, self.dt, self.u_prev_tmp, self.total_thrust_dot) 
        # self.x_prev_fmpc = obs

        # debugging vars
        # u_ref_nmpc = u_nmpc_horizon[:, 0]
        # u_prev = self.u0_prev
        # u_difference = u_prev - u_ref_nmpc

        # do controller 

        # MPC
        v = super().select_action(z_obs) # don't comment out, needed for trajectory stepping
        z_horizon = self.x_prev # set in linearMPC #8xN
        v_horizon = self.u_prev #2xN

        # Feedforward everything from NMPC - overwrite old stuff
        # v = v_nmpc_horizon[:, 0]
        # z_horizon = z_nmpc_horizon
        # v_horizon = v_nmpc_horizon

        # v = v_horizon[:, 1] # for logging
        
        
        # transform z and v to input u
        flat_action = _get_u_from_flat_states(z_horizon[:,1], v_horizon[:, 1]) # works
        # flat_action = _get_u_from_flat_states(z_horizon[:, 1], v) # also works but why???
        # flat_action = _get_u_from_flat_states(z_obs, v) # does not work


        # keep track of past actions
        self.u0_prev_prev = deepcopy(self.u0_prev)
        self.u0_prev = deepcopy(flat_action[0])

       
       # estimate total thrust derivative
        u0_dot = (self.u0_prev - self.u0_prev_prev)/self.dt

        u_horizon = np.zeros(np.shape(v_horizon))
        for i in range(self.T):
            u_horizon[:, i] = _get_u_from_flat_states(z_horizon[:,i], v_horizon[:,i])
        u_dot_central = (-u_horizon[:, 0]  + u_horizon[:, 2])/(2*self.dt)
       
          
        self.u0_dot = u_dot_central[0] 

        # action = u_nmpc_horizon[:, 0]
        action = flat_action

        # data logging
        self.results_dict['obs_x'].append(obs)
        self.results_dict['obs_z'].append(z_obs)
        self.results_dict['v'].append(v)
        self.results_dict['u'].append(flat_action)
        self.results_dict['T_dot'].append(self.u0_dot)
        self.results_dict['u_ref'].append(u_nmpc_horizon[:, 0])
        self.results_dict['horizon_v'].append(v_horizon)
        self.results_dict['horizon_z'].append(z_horizon)
        self.results_dict['horizon_u'].append(u_horizon)

        self.counter -=1
        if False: #self.counter <= 0:
            self.counter = 10
            # Plot states

            plot_data_comparison(z_horizon.transpose(), self.get_references().transpose(), range(self.T+1), 'Flat states over MPC horizon', 'Horizon time step')  

            plot_data_comparison(u_horizon.transpose(), u_nmpc_horizon.transpose(), range(self.T), 'Inputs U against NMPC reference', 'Horizon time step')  

            plt.show()

        return action
    
    
def _get_u_from_flat_states(z, v):
    beta_1 = 18.112984649321753
    beta_2 = 3.6800
    alpha_1 =  -140.8
    alpha_2 =  -13.4
    alpha_3 =  124.8
    g=9.81



    alpha = np.square(z[2]) + np.square(z[6]+g) # x_ddot^2 + (z_ddot+g)^2
    theta = np.arctan2(z[2], (z[6]+g))
    theta_dot = (z[3]*(z[6]+g)- z[2]*z[7])/alpha
    theta_ddot = 1/alpha * (v[0]*(z[6]+g) - z[2]*v[1]) + (1/np.square(alpha)) * (2*(z[6]+g)*z[7] + 2*z[2]*z[3]) * (z[2]*z[7] - z[3]*(z[6]+g))

    t = -(beta_2/beta_1) + np.sqrt(alpha)/beta_1
    p = (1/alpha_3) * (theta_ddot - alpha_1*theta -alpha_2*theta_dot)
    return np.array([t, p])

def _get_total_thrust_dot_from_flat_states(z):
    # currently not needed
    raise NotImplementedError

    alpha = np.square(z[2]) + np.square(z[6]+g) # x_ddot^2 + (z_ddot+g)^2
    t_dot = m*(z[2]*z[3] + (z[6]+g)*z[7])/np.sqrt(alpha)
    return t_dot

def _get_z_from_regular_states(x, u0, u0_dot):
    beta_1 = 18.112984649321753
    beta_2 = 3.6800
    g=9.81

    z = np.zeros(8)
    z[0] = x[0] # x
    z[1] = x[1] # x_dot
    z[2] = np.sin(x[4])*(beta_2 + beta_1*u0) # x_ddot
    z[3] = np.cos(x[4])*(beta_2 + beta_1*u0)*x[5] + np.sin(x[4])*beta_1*u0_dot # x_dddot
    z[4] = x[2] # z
    z[5] = x[3] # z_dot
    z[6] = np.cos(x[4])*(beta_2 + beta_1*u0)- g # z_ddot
    z[7] = -np.sin(x[4])*(beta_2 + beta_1*u0)*x[5] + np.cos(x[4])*beta_1*u0_dot# z_dddot
    return z

# not needed in FMPC, just to double check transformations
def _get_x_from_flat_states(z):
    g = 9.81


    x = np.zeros(6)
    x[0] = z[0]
    x[1] = z[1]
    x[2] = z[4]
    x[3] = z[5]
    x[4] = np.arctan2(z[2], (z[6]+g))
    x[5] = (z[3]*(z[6]+g)- z[2]*z[7])/((z[6]+g)**2 + z[2]**2)
    return x


def _get_z_from_regular_states_FD(x, x_prev, dt, u, t_tot_dot):
    m = 0.027
    g=9.81

    raise NotImplementedError # currently not needed

    # compute states_dot with finite differences to get x_ddot and z_ddot
    states_dot = (x-x_prev)/dt

    z = np.zeros(8)
    total_thrust = u[0]+u[1]
    total_thrust_dot = t_tot_dot #u_dot[0]+u_dot[1]
    z[0] = x[0] # x
    z[1] = x[1] # x_dot
    z[2] = states_dot[1] #(1/m)*(np.sin(x[4])*total_thrust) # x_ddot
    z[3] = (1/m)*(np.cos(x[4])*total_thrust*x[5] + np.sin(x[4])*total_thrust_dot)# x_dddot
    z[4] = x[2] # z
    z[5] = x[3] # z_dot
    z[6] = states_dot[3] #(1/m)*(np.cos(x[4])*total_thrust) - g # z_ddot
    z[7] = (1/m)*(np.sin(x[4])*total_thrust*x[5]*(-1) + np.cos(x[4])*total_thrust_dot) # z_dddot
    return z

def _create_flat_trajectory_circle():
    # constants
    scaling = 0.75
    traj_length = 6 # seconds
    sample_time = 0.02 # seconds, sampling time of trajectory, 50Hz
    traj_period = traj_length # one circle, not multiple
    z_offset = 1 # m above ground

    # create circle trajectory
    times = np.arange(0, traj_length + sample_time, sample_time)  # sample time added to make reference one step longer than traj_length
    ref_x = np.zeros((len(times)))
    ref_z = np.zeros((len(times)))
    ref_x_dot = np.zeros((len(times)))
    ref_z_dot = np.zeros((len(times)))
    ref_x_ddot = np.zeros((len(times)))
    ref_z_ddot = np.zeros((len(times)))
    ref_x_dddot = np.zeros((len(times)))
    ref_z_dddot = np.zeros((len(times)))
    ref_x_d4dot = np.zeros((len(times)))
    ref_z_d4dot = np.zeros((len(times)))

    traj_freq = 2.0 * np.pi / traj_period
    for index, t in enumerate(times):
        ref_x[index] = scaling * np.cos(traj_freq * t)
        ref_z[index] = scaling * np.sin(traj_freq * t) + z_offset
        ref_x_dot[index] = -scaling * traj_freq * np.sin(traj_freq * t)
        ref_z_dot[index] = scaling * traj_freq * np.cos(traj_freq * t)
        ref_x_ddot[index] = -scaling * traj_freq**2 * np.cos(traj_freq * t)
        ref_z_ddot[index] = -scaling * traj_freq**2 * np.sin(traj_freq * t)
        ref_x_dddot[index] = scaling * traj_freq**3 * np.sin(traj_freq * t)
        ref_z_dddot[index] = -scaling * traj_freq**3 * np.cos(traj_freq * t)
        #ref_x_d4dot[index] = scaling * traj_freq**4 * np.cos(traj_freq * t)
        #ref_z_d4dot[index] = scaling * traj_freq**4 * np.sin(traj_freq * t)

    z_traj = np.zeros((8,len(times)))
    z_traj[0, :] = ref_x[:]
    z_traj[1, :] = ref_x_dot[:]
    z_traj[2, :] = ref_x_ddot[:]
    z_traj[3, :] = ref_x_dddot[:]
    z_traj[4, :] = ref_z[:]
    z_traj[5, :] = ref_z_dot[:]
    z_traj[6, :] = ref_z_ddot[:]
    z_traj[7, :] = ref_z_dddot[:]

    return z_traj

def _load_flat_trajectory_circle(): # from NMPC
    with open('/home/tobias/Studium/masterarbeit/code/safe-control-gym/examples/mpc/temp-data/reference_NMPC.pkl', 'rb') as file:
        data_dict = pickle.load(file)
    z_ref = data_dict['z_ref']

    return z_ref.transpose()

