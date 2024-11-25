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

from termcolor import colored

from safe_control_gym.math_and_models.symbolic_systems import SymbolicModel

from plottingUtils import *
from safe_control_gym.utils.utils import timing

import time

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
        self.Q = get_cost_weight_matrix([50, 0.001, 0.1, 0.001, 50, 0.001, 0.1, 0.001], self.model.nx) 
        self.R = get_cost_weight_matrix([1e-6], self.model.nu)
        
        self.fs_obs = FlatStateObserver(self.env.INERTIAL_PROP, self.env.GRAVITY_ACC, self.dt, self.T)

        # open a csv file for controller time logging
        # self._time_log_file =  open(f'./{self.output_dir}/timing-data.csv', 'w+')
        # self._time_log_file =  open(f'./temp-data/timing-data.csv', 'w+')


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


    # overwrite to input flat trajectory into reference and initialize flat state observer
    def reset(self):
        '''Prepares for training or evaluation.'''
        # Setup reference input.
        if self.env.TASK == Task.STABILIZATION:
            self.mode = 'stabilization'
            self.x_goal = _get_flat_stabilization_goal(self.env.X_GOAL)            
            x_ini = self.env.__dict__['init_x'.upper()]
            z_ini = self.env.__dict__['init_z'.upper()]
            self.fs_obs.set_initial_hovering(x_ini, z_ini)
        elif self.env.TASK == Task.TRAJ_TRACKING:
            self.mode = 'tracking'
            #self.traj = self.env.X_GOAL.T
            # self.traj = _load_flat_trajectory_circle()
            # self.traj, initial_vals = _create_flat_trajectory_circle(self.env.TASK_INFO, self.env.EPISODE_LEN_SEC, self.dt, self.T, self.env.INERTIAL_PROP, self.env.GRAVITY_ACC)
            self.traj, initial_vals = _create_flat_trajectory_figure8(self.env.TASK_INFO, self.env.EPISODE_LEN_SEC, self.dt, self.T, self.env.INERTIAL_PROP, self.env.GRAVITY_ACC)
            # Step along the reference.
            self.traj_step = 0

            # initialize flat state observer in hovering
            x_ini = self.env.__dict__['init_x'.upper()]
            z_ini = self.env.__dict__['init_z'.upper()]
            self.fs_obs.set_initial_hovering(x_ini, z_ini)

            # # initialize the flat state observer by inputing u and a z and v reference horizon starting at timestep -1
            # z_horizon = initial_vals['z_ini_hrzn']
            # v_horizon = initial_vals['v_ini_hrzn']
            # u = initial_vals['u_ini'] 
            # self.fs_obs.input_FMPC_result(z_horizon, v_horizon, u)

            # x_ini = initial_vals['x_ini']
            # print(x_ini)

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
                             'ctrl_run_time': [],
                             }

    # @timing
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
        ts = time.time()    
        # get flat state estimation from observer
        z_obs = self.fs_obs.compute_observation(obs)
        
        # z_ref = self.get_references() # for debugging

        # run MPC controller 
        v = super().select_action(z_obs) 
        z_horizon = self.x_prev #8xN set in linearMPC
        v_horizon = self.u_prev #2xN       
        
        # flat input transformation: z and v to action u        
        action = _get_u_from_flat_states(z_horizon[:, 1], v_horizon[:, 0], self.env.INERTIAL_PROP, g=self.env.GRAVITY_ACC) 
        

        # feed data into observer
        self.fs_obs.input_FMPC_result(z_horizon, v_horizon, action)

        # data logging
        self.results_dict['obs_x'].append(obs)
        self.results_dict['obs_z'].append(z_obs)
        self.results_dict['v'].append(v)
        self.results_dict['u'].append(action)
        # self.results_dict['T_dot'].append(self.u0_dot)
        # self.results_dict['u_ref'].append(u_nmpc_horizon[:, 0])
        self.results_dict['horizon_v'].append(v_horizon)
        self.results_dict['horizon_z'].append(z_horizon)
        # self.results_dict['horizon_u'].append(u_horizon)

        # log execution time                
        te = time.time()
        self.results_dict['ctrl_run_time'].append(te-ts)
        
        return action
    
    # overwrite to change shape of goal states in stabilization task
    def get_references(self):
        '''Constructs reference states along mpc horizon.(nx, T+1).'''
        if self.env.TASK == Task.STABILIZATION:
            # Repeat goal state for horizon steps.
            goal_states = np.tile(_get_flat_stabilization_goal(self.env.X_GOAL).reshape(-1, 1), (1, self.T + 1))
        elif self.env.TASK == Task.TRAJ_TRACKING:
            # Slice trajectory for horizon steps, if not long enough, repeat last state.
            start = min(self.traj_step, self.traj.shape[-1])
            end = min(self.traj_step + self.T + 1, self.traj.shape[-1])
            remain = max(0, self.T + 1 - (end - start))
            goal_states = np.concatenate([
                self.traj[:, start:end],
                np.tile(self.traj[:, -1:], (1, remain))
            ], -1)
        else:
            raise Exception('Reference for this mode is not implemented.')
        return goal_states  # (nx, T+1).
    
class FlatStateObserver():
    def __init__(self,  inertial_prop, g, dt, horizon):
        self.inertial_prop = inertial_prop # alpha beta of model from env (from config file)
        self.GRAVITY = g
        self.dt = dt
        self.fmpc_horizon = horizon


    def set_initial_hovering(self, x_pos, z_pos):
        # initializes u, z and v horizon for a hovering state
        self.z_horizon = np.zeros([8, self.fmpc_horizon+1])
        self.v_horizon = np.zeros([2, self.fmpc_horizon])
        self.u = np.zeros(2)
        
        self.u[0] = (self.GRAVITY- self.inertial_prop['beta_2'])/self.inertial_prop['beta_1'] 
        # z_ini = _get_flat_stabilization_goal(x_obs)
        z_ini = np.zeros(8)
        z_ini[0] = x_pos
        z_ini[4] = z_pos
        for i in range(self.fmpc_horizon+1): # TODO make nicer with matrix repetition
            self.z_horizon[:, i] = z_ini

        
    def input_FMPC_result(self, z_horizon, v_horizon, u):
        # just save them away
        self.z_horizon = z_horizon
        self.v_horizon = v_horizon
        self.u = u     

              
    def compute_observation(self, x_obs): 
        # estimate u_dot at current time step, based on z_horizon and v_horizon set in last time step
        u_comp_length = 3
        u_horizon = np.zeros([2, u_comp_length])
        for i in range(u_comp_length):
            u_horizon[:, i] = _get_u_from_flat_states(self.z_horizon[:,i], self.v_horizon[:,i], self.inertial_prop, self.GRAVITY)

        u_dot_central = (-u_horizon[:, 0]  + u_horizon[:, 2])/(2*self.dt)
       
        u0_dot = u_dot_central[0]

        # state estimation using system dynamics
        z_obs = _get_z_from_regular_states(x_obs, self.u[0], u0_dot, self.inertial_prop, self.GRAVITY) 
        return z_obs
       
    
def _get_u_from_flat_states(z, v, dyn_pars, g):
    beta_1 = dyn_pars['beta_1']
    beta_2 = dyn_pars['beta_2']
    alpha_1 =  dyn_pars['alpha_1']
    alpha_2 =  dyn_pars['alpha_2']
    alpha_3 =  dyn_pars['alpha_3']

    # print(colored('WARNING: Flat transformation: model parameters not provided, using defaults!', 'yellow'))
    # beta_1 = 18.112984649321753
    # beta_2 = 3.6800
    # alpha_1 =  -140.8
    # alpha_2 =  -13.4
    # alpha_3 =  124.8

    term_acc_sqrd = (z[2])**2 + (z[6]+g)**2 # x_ddot^2 + (z_ddot+g)^2
    theta = np.arctan2(z[2], (z[6]+g))
    theta_dot = (z[3]*(z[6]+g)- z[2]*z[7])/term_acc_sqrd
    theta_ddot = 1/term_acc_sqrd * (v[0]*(z[6]+g) - z[2]*v[1]) + (1/(term_acc_sqrd**2)) * (2*(z[6]+g)*z[7] + 2*z[2]*z[3]) * (z[2]*z[7] - z[3]*(z[6]+g))

    t = -(beta_2/beta_1) + np.sqrt(term_acc_sqrd)/beta_1
    p = (1/alpha_3) * (theta_ddot - alpha_1*theta -alpha_2*theta_dot)
    return np.array([t, p])

def _get_total_thrust_dot_from_flat_states(z):
    # currently not needed
    raise NotImplementedError

    alpha = np.square(z[2]) + np.square(z[6]+g) # x_ddot^2 + (z_ddot+g)^2
    t_dot = m*(z[2]*z[3] + (z[6]+g)*z[7])/np.sqrt(alpha)
    return t_dot

def _get_z_from_regular_states(x, u0, u0_dot, dyn_pars, g):    
   
    beta_1 = dyn_pars['beta_1']
    beta_2 = dyn_pars['beta_2']        

    # print(colored('WARNING: Flat transformation: model parameters not provided, using defaults!', 'yellow'))
    # beta_1 = 18.112984649321753
    # beta_2 = 3.6800
        
    z = np.zeros(8)
    sin_theta = np.sin(x[4])
    cos_theta = np.cos(x[4])

    z[0] = x[0] # x
    z[1] = x[1] # x_dot
    z[2] = sin_theta*(beta_2 + beta_1*u0) # x_ddot    
    z[3] = cos_theta*(beta_2 + beta_1*u0)*x[5] + sin_theta*beta_1*u0_dot # x_dddot
    z[4] = x[2] # z
    z[5] = x[3] # z_dot
    z[6] = cos_theta*(beta_2 + beta_1*u0)- g # z_ddot
    z[7] = -sin_theta*(beta_2 + beta_1*u0)*x[5] + cos_theta*beta_1*u0_dot# z_dddot
    return z

# not needed in FMPC, used for x_ini in trajectory generation
def _get_x_from_flat_states(z, g):
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

def _get_flat_stabilization_goal(x_stab):
    z = np.zeros(8)
    z[0] = x_stab[0]
    z[1] = x_stab[1]
    z[4] = x_stab[2]
    z[5] = x_stab[3]
    return z

def _create_flat_trajectory_circle(task_info, traj_length, sample_time, horizon, inertial_prop, gravity):
    # task info parameters from yaml file
    traj_scaling = task_info.trajectory_scale
    num_cycles = task_info.num_cycles
    pos_offset = task_info.trajectory_position_offset

    traj_period = traj_length/num_cycles     

    # create circle trajectory
    times = np.arange(0, traj_length + sample_time*(1+horizon), sample_time)  # sample time added to make reference one step longer than traj_length
    z_traj = np.zeros((8,len(times)))
    v_traj = np.zeros((2,len(times)))

    def circle_traj_at_t(t, scaling, freq, offset):
        z = np.zeros(8)
        v = np.zeros(2)
        z[0] = scaling * np.cos(freq * t) + offset[0] # x
        z[1] = -scaling * freq * np.sin(freq * t) # x_dot
        z[2] = -scaling * freq**2 * np.cos(freq * t) # x_ddot
        z[3] = scaling * freq**3 * np.sin(freq * t) # x_dddot

        v[0] = scaling * freq**4 * np.cos(freq * t) # x_d4dot = v0

        z[4] = scaling * np.sin(freq * t) + offset[1]        
        z[5] = scaling * freq * np.cos(freq * t)        
        z[6] = -scaling * freq**2 * np.sin(freq * t)        
        z[7] = -scaling * freq**3 * np.cos(freq * t)

        v[1] = scaling * freq**4 * np.sin(freq * t)
        return z, v

    traj_freq = 2.0 * np.pi / traj_period
    for index, t in enumerate(times):
        z_traj[:, index], v_traj[:, index] = circle_traj_at_t(t, traj_scaling, traj_freq, pos_offset)

    # calculate initial values for x, u and u_dot for flat state observer
    initial_vals = {}
    z_minus1, v_minus1 = circle_traj_at_t(-sample_time, traj_scaling, traj_freq, pos_offset)
    z_ini_horizon = np.hstack((np.array([z_minus1]).T, z_traj[:, 0:horizon-1]))
    v_ini_horizon = np.concatenate((np.array([v_minus1]).T, v_traj[:, 0:horizon-1]), axis=1)

    u_ini = _get_u_from_flat_states(z_ini_horizon[:, 1], v_ini_horizon[:, 0], inertial_prop, gravity) #NOTE: the indices need to match the FMPC implementation
    x_ini = _get_x_from_flat_states(z_traj[:, 0], gravity)

    initial_vals['z_ini_hrzn'] = z_ini_horizon
    initial_vals['v_ini_hrzn'] = v_ini_horizon
    initial_vals['u_ini'] = u_ini
    initial_vals['x_ini'] = x_ini

    # save reference trajectory for evaluation
    data_dict = {'z_ref': z_traj, 'v_ref':v_traj}
    with open('/home/tobias/Studium/masterarbeit/code/safe-control-gym/examples/mpc/temp-data/reference_analytic.pkl', 'wb') as file_ref:
        pickle.dump(data_dict, file_ref)
    
    return z_traj, initial_vals

def _create_flat_trajectory_figure8(task_info, traj_length, sample_time, horizon, inertial_prop, gravity):
    # task info parameters from yaml file
    traj_scaling = task_info.trajectory_scale
    num_cycles = task_info.num_cycles
    pos_offset = task_info.trajectory_position_offset

    traj_period = traj_length/num_cycles     

    # create circle trajectory
    times = np.arange(0, traj_length + sample_time*(1+horizon), sample_time)  
    z_traj = np.zeros((8,len(times)))
    v_traj = np.zeros((2,len(times)))

    def figure8_traj_at_t(t, scaling, freq, offset):
        z = np.zeros(8)
        v = np.zeros(2)
        z[0] = scaling * np.sin(freq * t) + offset[0] # x
        z[1] = scaling * freq * np.cos(freq * t) # x_dot
        z[2] = -scaling * freq**2 * np.sin(freq * t) # x_ddot
        z[3] = -scaling * freq**3 * np.cos(freq * t) # x_dddot

        v[0] = scaling * freq**4 * np.sin(freq * t) # x_d4dot = v0

        z[4] = scaling * np.sin(freq * t) *np.cos(freq * t) + offset[1]        
        z[5] = scaling * freq * (np.cos(freq * t)**2 - np.sin(freq*t)**2)       
        z[6] = -scaling * freq**2 * 4 * np.sin(freq * t) *np.cos(freq * t)        
        z[7] = scaling * freq**3 * 4 * (np.sin(freq * t)**2 - np.cos(freq*t)**2)

        v[1] = scaling * freq**4 * 16 * (np.sin(freq * t) *np.cos(freq * t))
        return z, v

    traj_freq = 2.0 * np.pi / traj_period
    for index, t in enumerate(times):
        z_traj[:, index], v_traj[:, index] = figure8_traj_at_t(t, traj_scaling, traj_freq, pos_offset)

    # calculate initial values for x, u and u_dot for flat state observer
    initial_vals = {}
    z_minus1, v_minus1 = figure8_traj_at_t(-sample_time, traj_scaling, traj_freq, pos_offset)
    z_ini_horizon = np.hstack((np.array([z_minus1]).T, z_traj[:, 0:horizon-1]))
    v_ini_horizon = np.concatenate((np.array([v_minus1]).T, v_traj[:, 0:horizon-1]), axis=1)

    u_ini = _get_u_from_flat_states(z_ini_horizon[:, 1], v_ini_horizon[:, 0], inertial_prop, gravity) #NOTE: the indices need to match the FMPC implementation
    x_ini = _get_x_from_flat_states(z_traj[:, 0], gravity)

    initial_vals['z_ini_hrzn'] = z_ini_horizon
    initial_vals['v_ini_hrzn'] = v_ini_horizon
    initial_vals['u_ini'] = u_ini
    initial_vals['x_ini'] = x_ini

    # save reference trajectory for evaluation
    data_dict = {'z_ref': z_traj, 'v_ref':v_traj}
    with open('/home/tobias/Studium/masterarbeit/code/safe-control-gym/examples/mpc/temp-data/reference_analytic.pkl', 'wb') as file_ref:
        pickle.dump(data_dict, file_ref)
    
    return z_traj, initial_vals

def _load_flat_trajectory_circle(): # from NMPC
    raise NotImplementedError # initial values for state estimator not returned yet
    with open('/home/tobias/Studium/masterarbeit/code/safe-control-gym/examples/mpc/temp-data/reference_NMPC.pkl', 'rb') as file:
        data_dict = pickle.load(file)
    z_ref = data_dict['z_ref']



    return z_ref.transpose()

