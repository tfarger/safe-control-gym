import numpy as np
import matplotlib.pyplot as plt
import pickle

# variables, get from env later
Iyy = 1.4e-5
l = 0.0397
m = 0.027
g=9.81

def get_x_from_flat_states(z):
    x = np.zeros(6)
    x[0] = z[0]
    x[1] = z[1]
    x[2] = z[4]
    x[3] = z[5]
    x[4] = np.arctan2(z[2], (z[6]+g))
    x[5] = (z[3]*(z[6]+g)- z[2]*z[7])/((z[6]+g)**2 + z[2]**2)
    return x

def get_u_from_flat_states(z, v):
    alpha = np.square(z[2]) + np.square(z[6]+g) # x_ddot^2 + (z_ddot+g)^2
    theta_ddot = 1/alpha * (v[0]*(z[6]+g) - z[2]*v[1]) + (1/np.square(alpha)) * (2*(z[6]+g)*z[7] + 2*z[2]*z[3]) * (z[2]*z[7] - z[3]*(z[6]+g))

    t1 = 0.5*(m*np.sqrt(alpha) - theta_ddot*Iyy*np.sqrt(2)/l)
    t2 = 0.5*(m*np.sqrt(alpha) + theta_ddot*Iyy*np.sqrt(2)/l)
    return [t1, t2]

def get_z_from_regular_states(x, u, t_tot_dot):
    z = np.zeros(8)
    total_thrust = u[0]+u[1]
    total_thrust_dot = t_tot_dot #u_dot[0]+u_dot[1]
    z[0] = x[0] # x
    z[1] = x[1] # x_dot
    z[2] = (1/m)*(np.sin(x[4])*total_thrust) # x_ddot
    z[3] = (1/m)*(np.cos(x[4])*total_thrust*x[5] + np.sin(x[4])*total_thrust_dot)# x_dddot
    z[4] = x[2] # z
    z[5] = x[3] # z_dot
    z[6] = (1/m)*(np.cos(x[4])*total_thrust) - g # z_ddot
    z[7] = (1/m)*(np.sin(x[4])*total_thrust*x[5]*(-1) + np.cos(x[4])*total_thrust_dot) # z_dddot
    return z

def get_total_thrust_dot_from_flat_states(z):
    alpha = np.square(z[2]) + np.square(z[6]+g) # x_ddot^2 + (z_ddot+g)^2
    t_dot = m*(z[2]*z[3] + (z[6]+g)*z[7])/np.sqrt(alpha)
    return t_dot

def plot_z_trajectory(z, t):
    plt.figure()
    plt.subplot(811)
    plt.plot(t, z[0, :])
    plt.ylabel('x')

    plt.subplot(812)
    plt.plot(t, z[1, :])
    plt.ylabel('x_dot')

    plt.subplot(813)
    plt.plot(t, z[2, :])
    plt.ylabel('x_ddot')

    plt.subplot(814)
    plt.plot(t, z[3, :])
    plt.ylabel('x_dddot')

    plt.subplot(815)
    plt.plot(t, z[4, :])
    plt.ylabel('z')

    plt.subplot(816)
    plt.plot(t, z[5, :])
    plt.ylabel('z_dot')

    plt.subplot(817)
    plt.plot(t, z[6, :])
    plt.ylabel('z_ddot')
  
    plt.subplot(818)
    plt.plot(t, z[7, :])
    plt.ylabel('z_dddot')

    plt.xlabel('time in s')
    #plt.show()
    return True

def plot_x_trajectory(x, t):
    plt.figure()
    plt.subplot(611)
    plt.plot(t, x[0, :])
    plt.ylabel('x')

    plt.subplot(612)
    plt.plot(t, x[1, :])
    plt.ylabel('x_dot')

    plt.subplot(613)
    plt.plot(t, x[2, :])
    plt.ylabel('z')

    plt.subplot(614)
    plt.plot(t, x[3, :])
    plt.ylabel('z_dot')

    plt.subplot(615)
    plt.plot(t, x[4, :])
    plt.ylabel('theta')

    plt.subplot(616)
    plt.plot(t, x[5, :])
    plt.ylabel('theta_dot')

    plt.xlabel('time in s')
    #plt.show()
    return True

def plot_u_trajectory(u, t):
    plt.figure()
    plt.subplot(211)
    plt.plot(t, u[0, :])
    plt.ylabel('T1')

    plt.subplot(212)
    plt.plot(t, u[1, :])
    plt.ylabel('T2')

    plt.xlabel('time in s')
    #plt.show()
    return True




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
    ref_z_ddot[index] = -scaling * traj_freq**2 * np.sin(traj_freq * t) # - g # do not add gravity here, otherwise drone will do a flip/looping
    ref_x_dddot[index] = scaling * traj_freq**3 * np.sin(traj_freq * t)
    ref_z_dddot[index] = -scaling * traj_freq**3 * np.cos(traj_freq * t)
    ref_x_d4dot[index] = scaling * traj_freq**4 * np.cos(traj_freq * t)
    ref_z_d4dot[index] = scaling * traj_freq**4 * np.sin(traj_freq * t)

# print(ref_x, ref_x_dot, ref_x_ddot, ref_x_dddot, ref_x_d4dot)

################################################################################################
# # check transformations forward and backwards --> same result
# for obs_index in range(len(times)):
#     # generate one observation: flat states at a timestep
#     #obs_index = 0
#     obs_z = [ref_x[obs_index], ref_x_dot[obs_index], ref_x_ddot[obs_index], ref_x_dddot[obs_index], ref_z[obs_index], ref_z_dot[obs_index], ref_z_ddot[obs_index], ref_z_dddot[index]]
#     obs_v = [ref_x_d4dot[obs_index], ref_z_d4dot[obs_index]]
    
#     print('======================================================')
#     print(obs_z)
#     # print(obs_v)

#     x_obs = get_x_from_flat_states(obs_z)
#     u_obs = get_u_from_flat_states(obs_z, obs_v)
#     thrust_tot_dot = get_total_thrust_dot_from_flat_states(obs_z)

#     # print('======================')
#     print(x_obs)
#     # print(u_obs)
#     # print(thrust_tot_dot)

#     z_restored = get_z_from_states(x_obs, u_obs, thrust_tot_dot)

#     # print('==============')
#     # print(z_restored)

#     #print((obs_z-z_restored))

########################################################################################
# plot trajectory

z_traj = np.zeros((8,len(times)))
z_traj[0, :] = ref_x[:]
z_traj[1, :] = ref_x_dot[:]
z_traj[2, :] = ref_x_ddot[:]
z_traj[3, :] = ref_x_dddot[:]
z_traj[4, :] = ref_z[:]
z_traj[5, :] = ref_z_dot[:]
z_traj[6, :] = ref_z_ddot[:]
z_traj[7, :] = ref_z_dddot[:]

#plot_z_trajectory(z_traj, times)

x_traj = np.zeros((6,len(times)))
for index in range(len(times)):
    x_traj[:, index] = get_x_from_flat_states(z_traj[:, index])

#plot_x_trajectory(x_traj, times)

#######################################################################################

# load trajectory data from NMPC
traj_sample_time = sample_time # 50Hz for measurement

with open('./examples/mpc/temp-data/mpc_data_quadrotor_traj_tracking.pkl', 'rb') as file:
    data_dict = pickle.load(file)
    data_dict = data_dict['trajs_data']
    #print(data_dict.keys())

states_all = data_dict['state'][0]
actions_all = data_dict['action'][0]
#print(np.shape(actions))

# get the third circle at 50Hz sampling frequency
start_index = 600 
stop_index = 901

x_real_traj = states_all[start_index:stop_index, :]
u_real_traj = actions_all[start_index:stop_index, :]

plot_x_trajectory(x_real_traj.transpose(), times)
# plot_u_trajectory(u_real_traj.transpose(), times)

# approximate u_dot with finite differences
u_tmp = actions_all[start_index:stop_index+1, :]
u_dot = np.zeros(np.shape(u_real_traj))
for i in range(np.shape(u_real_traj)[0]):
    u_dot[i, :] = (u_tmp[i+1, :] - u_tmp[i, :])/traj_sample_time

#plot_u_trajectory(u_dot.transpose(), times)

# build z vector from this
z_real_traj = np.zeros((8, np.shape(x_real_traj)[0]))
for i in range(np.shape(x_real_traj)[0]):
    z_real_traj[:, i] = get_z_from_regular_states(x_real_traj[i, :].transpose(), u_real_traj[i, :].transpose(), u_dot[i, 0]+ u_dot[i, 1])

plot_z_trajectory(z_real_traj, times)

# double check, transform back to x

x_traj_recovered = np.zeros((6,len(times)))
for index in range(len(times)):
    x_traj_recovered[:, index] = get_x_from_flat_states(z_real_traj[:, index])

plot_x_trajectory(x_traj_recovered, times)
print(max(x_real_traj.transpose()-x_traj_recovered))

plt.show()

