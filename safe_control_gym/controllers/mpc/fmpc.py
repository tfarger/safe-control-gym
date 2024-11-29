'''Flatness based Model Predictive Control.

Based on:
    Linear MPC 
'''
#TODO s
# reenable state constraints in linearMPC


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
            q_mpc=[1],
            r_mpc=[1],
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

        self.inertial_prop = {} # not as a nice variable in the env yet, thats why its defined here again
        self.inertial_prop['alpha_0'] = 20.907574256269616
        self.inertial_prop['alpha_1'] = 3.653687545690674
        self.inertial_prop['beta_0'] = -130.3
        self.inertial_prop['beta_1'] = -16.33
        self.inertial_prop['beta_2'] = 119.3
        self.inertial_prop['gamma_0'] = -99.94
        self.inertial_prop['gamma_1'] = -13.3
        self.inertial_prop['gamma_2'] = 84.73

        # TODO: setup environment equilibrium
        # self.X_EQ = np.atleast_2d(self.env.X_GOAL)[0,:].T
        # self.U_EQ = np.atleast_2d(self.env.U_GOAL)[0,:]

        self.X_EQ = np.atleast_2d(self.model.X_EQ)[0, :].T
        self.U_EQ = np.atleast_2d(self.model.U_EQ)[0, :].T 
        assert solver in ['qpoases', 'qrqp', 'sqpmethod', 'ipopt'], '[Error]. MPC Solver not supported.'
        self.solver = solver

        # overwrite definitions in parent init function to fit flat model
        self.Q = get_cost_weight_matrix(q_mpc, self.model.nx) 
        self.R = get_cost_weight_matrix(r_mpc, self.model.nu)
        
        # self.fs_obs = FlatStateObserver(self.env.INERTIAL_PROP, self.env.GRAVITY_ACC, self.dt, self.T)
        self.fs_obs = FlatStateObserver(self.inertial_prop, self.env.GRAVITY_ACC, self.dt, self.T)



    def _setup_flat_model_symbolic(self):
        '''Creates symbolic (CasADi) models for dynamics, observation, and cost.

        Args:
            prior_prop (dict): specify the prior inertial prop to use in the symbolic model.
        ''' 
        nx, nu = 12, 3
      
        dt = self.dt
        # Define states.
        z = cs.MX.sym('z')
        z_dot = cs.MX.sym('z_dot')
        z_ddot = cs.MX.sym('z_ddot')   
        z_dddot = cs.MX.sym('z_dddot')  

        y = cs.MX.sym('y')
        y_dot = cs.MX.sym('y_dot')
        y_ddot = cs.MX.sym('y_ddot')   
        y_dddot = cs.MX.sym('y_dddot')       
        
        x = cs.MX.sym('x')
        x_dot = cs.MX.sym('x_dot')
        x_ddot = cs.MX.sym('x_ddot')
        x_dddot = cs.MX.sym('x_dddot')
        
        X = cs.vertcat(x, x_dot, x_ddot, x_dddot, y, y_dot, y_ddot, y_dddot, z, z_dot, z_ddot, z_dddot)
        # Define flat inputs 
        v1 = cs.MX.sym('v1')
        v2 = cs.MX.sym('v2')
        v3 = cs.MX.sym('v3')
        U = cs.vertcat(v1, v2, v3)
        # Define dynamics equations.
        X_dot = cs.vertcat(x_dot, x_ddot, x_dddot, v1,
                           y_dot, y_ddot, y_dddot, v2,
                           z_dot, z_ddot, z_dddot, v3)
        # Define observation.
        Y = cs.vertcat(x, x_dot, x_ddot, x_dddot, y, y_dot, y_ddot, y_dddot, z, z_dot, z_ddot, z_dddot)
        
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
            y_ini = self.env.__dict__['init_y'.upper()]
            z_ini = self.env.__dict__['init_z'.upper()]
            self.fs_obs.set_initial_hovering(x_ini, y_ini, z_ini)
        elif self.env.TASK == Task.TRAJ_TRACKING:
            self.mode = 'tracking'
            #self.traj = self.env.X_GOAL.T
            # self.traj = _load_flat_trajectory_circle()
            # self.traj, initial_vals = _create_flat_trajectory_circle(self.env.TASK_INFO, self.env.EPISODE_LEN_SEC, self.dt, self.T, self.env.INERTIAL_PROP, self.env.GRAVITY_ACC)
            self.traj, initial_vals = _create_flat_trajectory_figure8(self.env.TASK_INFO, self.env.EPISODE_LEN_SEC, self.dt, self.T, self.inertial_prop, self.env.GRAVITY_ACC)
            # Step along the reference.
            self.traj_step = 0

            # initialize flat state observer in hovering
            x_ini = self.env.__dict__['init_x'.upper()]
            y_ini = self.env.__dict__['init_y'.upper()]
            z_ini = self.env.__dict__['init_z'.upper()]
            self.fs_obs.set_initial_hovering(x_ini, y_ini, z_ini)

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
        
        z_ref = self.get_references() # for debugging

        # run MPC controller 
        v = super().select_action(z_obs) 
        z_horizon = self.x_prev #8xN set in linearMPC
        v_horizon = self.u_prev #2xN       
        
        # flat input transformation: z and v to action u        
        action = _get_u_from_flat_states(z_horizon[:, 1], v_horizon[:, 0], self.inertial_prop, g=self.env.GRAVITY_ACC) 
        

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


    def set_initial_hovering(self, x_pos, y_pos, z_pos):
        # initializes u, z and v horizon for a hovering state
        self.z_horizon = np.zeros([12, self.fmpc_horizon+1])
        self.v_horizon = np.zeros([3, self.fmpc_horizon])
        self.u = np.zeros(2)
        
        self.u[0] = (self.GRAVITY- self.inertial_prop['alpha_1'])/self.inertial_prop['alpha_0'] 
        # z_ini = _get_flat_stabilization_goal(x_obs)
        z_ini = np.zeros(12)
        z_ini[0] = x_pos
        z_ini[4] = y_pos
        z_ini[8] = z_pos
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
        u_horizon = np.zeros([3, u_comp_length])
        for i in range(u_comp_length):
            u_horizon[:, i] = _get_u_from_flat_states(self.z_horizon[:,i], self.v_horizon[:,i], self.inertial_prop, self.GRAVITY)

        u_dot_central = (-u_horizon[:, 0]  + u_horizon[:, 2])/(2*self.dt)
       
        u0_dot = u_dot_central[0]

        # state estimation using system dynamics
        z_obs = _get_z_from_regular_states(x_obs, self.u[0], u0_dot, self.inertial_prop, self.GRAVITY) 
        return z_obs
       
    
def _get_u_from_flat_states(z, v, dyn_pars, g):
    
    alpha_0 = dyn_pars['alpha_0'] 
    alpha_1 = dyn_pars['alpha_1'] 
    beta_0 = dyn_pars['beta_0'] 
    beta_1 = dyn_pars['beta_1'] 
    beta_2 = dyn_pars['beta_2'] 
    gamma_0 = dyn_pars['gamma_0'] 
    gamma_1 = dyn_pars['gamma_1'] 
    gamma_2 = dyn_pars['gamma_2']

    term_xz_acc_sqrd = (z[2])**2 + (z[10]+g)**2 # x_ddot^2 + (z_ddot+g)^2
    theta = np.arctan2(z[2], (z[10]+g))
    theta_dot = (z[3]*(z[10]+g)- z[2]*z[11])/term_xz_acc_sqrd
    theta_ddot = 1/term_xz_acc_sqrd * (v[0]*(z[10]+g) - z[2]*v[2]) + (1/(term_xz_acc_sqrd**2)) * (2*(z[10]+g)*z[11] + 2*z[2]*z[3]) * (z[2]*z[11] - z[3]*(z[10]+g))

    phi = np.arctan2(-z[6], np.sqrt(term_xz_acc_sqrd))
    phi_dot = ((((g + z[10])*z[11] + z[2]*z[3])*z[6]) - ((g + z[10])**2 + z[2]**2)*z[7])/(np.sqrt((g + z[10])**2 + z[2]**2)*((g + z[10])**2 + z[2]**2 + z[6]**2))
    phi_ddot = ((-((g + z[10])*z[11] + z[2]*z[3])*(((g + z[10])*z[11] + z[2]*z[3])*z[6] - ((g + z[10])**2 + z[2]**2)*z[7])*((g + z[10])**2 + z[2]**2 + z[6]**2) - 2*(((g + z[10])*z[11] + z[2]*z[3])*z[6] - ((g + z[10])**2 + z[2]**2)*z[7])*((g + z[10])**2 + z[2]**2)*((g + z[10])*z[11] + z[2]*z[3] + z[6]*z[7]) + ((g + z[10])**2 + z[2]**2)*(-((g + z[10])*z[11] + z[2]*z[3])*z[7] - ((g + z[10])**2 + z[2]**2)*v[1] + ((g + z[10])*v[2] + z[2]*v[0] + z[3]**2 + z[11]**2)*z[6])*((g + z[10])**2 + z[2]**2 + z[6]**2)))/((((g + z[10])**2 + z[2]**2)**(3/2))*((g + z[10])**2 + z[2]**2 + z[6]**2)**2)

    
    t = (1/alpha_0)*(np.sqrt(z[2]**2 + z[6]**2 + (z[10]+g)**2)-alpha_1)
    r = (1/beta_2) * (phi_ddot - beta_0*phi -beta_1*phi_dot)
    p = (1/gamma_2) * (theta_ddot - gamma_0*theta -gamma_1*theta_dot)
    return np.array([t, r, p])

def _get_total_thrust_dot_from_flat_states(z):
    # currently not needed
    raise NotImplementedError

    alpha = np.square(z[2]) + np.square(z[6]+g) # x_ddot^2 + (z_ddot+g)^2
    t_dot = m*(z[2]*z[3] + (z[6]+g)*z[7])/np.sqrt(alpha)
    return t_dot

def _get_z_from_regular_states(x, u0, u0_dot, dyn_pars, g):    
   
    alpha_0 = dyn_pars['alpha_0'] 
    alpha_1 = dyn_pars['alpha_1'] 
    # beta_0 = dyn_pars['beta_0'] 
    # beta_1 = dyn_pars['beta_1'] 
    # beta_2 = dyn_pars['beta_2'] 
    # gamma_0 = dyn_pars['gamma_0'] 
    # gamma_1 = dyn_pars['gamma_1'] 
    # gamma_2 = dyn_pars['gamma_2']       

    # print(colored('WARNING: Flat transformation: model parameters not provided, using defaults!', 'yellow'))
    # beta_1 = 18.112984649321753
    # beta_2 = 3.6800
        
    z = np.zeros(12)
    sin_theta = np.sin(x[7])
    cos_theta = np.cos(x[7])
    sin_phi = np.sin(x[6])
    cos_phi = np.cos(x[6])
    (alpha_0*u0 + alpha_1)

    z[0] = x[0] # x
    z[1] = x[1] # x_dot
    z[2] = sin_theta*cos_phi*(alpha_0*u0 + alpha_1) # x_ddot    
    z[3] = -sin_phi*sin_theta*(alpha_0*u0 + alpha_1)*x[8] + cos_phi*cos_theta*(alpha_0*u0 + alpha_1)*x[9] + cos_phi*sin_theta*alpha_0*u0_dot # x_dddot
    z[4] = x[2] # y
    z[5] = x[3] # y_dot
    z[6] = -sin_phi*(alpha_0*u0 + alpha_1) # y_ddot
    z[7] = -cos_phi*(alpha_0*u0 + alpha_1)*x[8] - sin_phi*alpha_0*u0_dot # y_dddot
    z[8] = x[4] # y
    z[9] = x[5] # y_dot
    z[10] = cos_theta*cos_phi*(alpha_0*u0 + alpha_1)- g # y_ddot
    z[11] = -sin_phi*cos_theta*(alpha_0*u0 + alpha_1)*x[8] - cos_phi*sin_theta*(alpha_0*u0 + alpha_1)*x[9] + cos_phi*cos_theta*alpha_0*u0_dot # z_dddot
    return z

# not needed in FMPC, used for x_ini in trajectory generation
def _get_x_from_flat_states(z, g):
    # raise NotImplementedError
    term_xz_acc_sqrd = (z[2])**2 + (z[10]+g)**2 # x_ddot^2 + (z_ddot+g)^2
    theta = np.arctan2(z[2], (z[10]+g))
    theta_dot = (z[3]*(z[10]+g)- z[2]*z[11])/term_xz_acc_sqrd

    phi = np.arctan2(-z[6], np.sqrt(term_xz_acc_sqrd))
    phi_dot = ((((g + z[10])*z[11] + z[2]*z[3])*z[6]) - ((g + z[10])**2 + z[2]**2)*z[7])/(np.sqrt((g + z[10])**2 + z[2]**2)*((g + z[10])**2 + z[2]**2 + z[6]**2))

    x = np.zeros(10)
    x[0] = z[0]
    x[1] = z[1]
    x[2] = z[4]
    x[3] = z[5]
    x[4] = z[8]
    x[5] = z[9]
    x[6] = phi
    x[7] = theta
    x[8] = phi_dot
    x[9] = theta_dot
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
    # z[1] = x_stab[1]
    z[4] = x_stab[2]
    # z[5] = x_stab[3]
    z[8] = x_stab[4]
    # z[9] = x_stab[5]
    return z

def _create_flat_trajectory_circle(task_info, traj_length, sample_time, horizon, inertial_prop, gravity):
    raise NotImplementedError
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
    z_traj = np.zeros((12,len(times)))
    v_traj = np.zeros((3,len(times)))

    def figure8_traj_at_t(t, scaling, freq, offset):
        z = np.zeros(12)
        v = np.zeros(3)
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

        # z values are all = 1 for figure 8 in xy plane
        z[8] = 1.0

        return z, v

    traj_freq = 2.0 * np.pi / traj_period
    for index, t in enumerate(times):
        z_traj[:, index], v_traj[:, index] = figure8_traj_at_t(t, traj_scaling, traj_freq, pos_offset)

    # calculate initial values for x, u and u_dot for flat state observer
    initial_vals = {}
    # z_minus1, v_minus1 = figure8_traj_at_t(-sample_time, traj_scaling, traj_freq, pos_offset)
    # z_ini_horizon = np.hstack((np.array([z_minus1]).T, z_traj[:, 0:horizon-1]))
    # v_ini_horizon = np.concatenate((np.array([v_minus1]).T, v_traj[:, 0:horizon-1]), axis=1)

    # u_ini = _get_u_from_flat_states(z_ini_horizon[:, 1], v_ini_horizon[:, 0], inertial_prop, gravity) #NOTE: the indices need to match the FMPC implementation
    # x_ini = _get_x_from_flat_states(z_traj[:, 0], gravity)

    # initial_vals['z_ini_hrzn'] = z_ini_horizon
    # initial_vals['v_ini_hrzn'] = v_ini_horizon
    # initial_vals['u_ini'] = u_ini
    # initial_vals['x_ini'] = x_ini

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

