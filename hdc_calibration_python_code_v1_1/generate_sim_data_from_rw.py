from pybullet_environment import PybulletEnvironment
from tqdm import tqdm
import numpy as np
import pickle
import scipy.spatial.transform.rotation as R
from scipy import interpolate
import math
from helper import angleDist
import matplotlib.pyplot as plt
import tikzplotlib

#Recorded imu data messages from ROS Node from IMU
#Recorded IMU data messages from ROS Node from Kinect
#--> Into ROS.bag file
#--> Convert it to csv file, and import it into a readable picklefile to run the model with data."""


def normalize_angle_positive(angle):
    """
    Wrap the angle between 0 and 2 * pi.
    """
    pi_2 = 2. * np.pi

    return math.fmod(math.fmod(angle, pi_2) + pi_2, pi_2)

# robot timestep
agentState =  {'Position': None,'Orientation': None}
thetas = []
agentStates = []
dt_robot = 0.05
rw_run = "circle"

# read data from csv-files
ors_data_in = np.genfromtxt('data/rw_data/'+rw_run+'2_or.csv', delimiter=',', skip_header=1)[:, [0, 4, 5, 6, 7]]  # timestamp, quaternions(x,y,z,w), (av would be column 19)
pos_data_in = np.genfromtxt('data/rw_data/'+rw_run+'2_pos.csv', delimiter=',', skip_header=1)[:, [0, 1, 2]] # timestamp, x-pos, y-pos
outfile = "data/sim/sim_data_rw_"+rw_run+".p"

# rescale timesteps
ors_data_in[:, [0]] = ors_data_in[:, [0]] - ors_data_in[[0], [0]]
ors_data_in[:, [0]] = ors_data_in[:, [0]] / 1E9

pos_data_in[:, [0]] = pos_data_in[:, [0]] - pos_data_in[[0], [0]]
pos_data_in[:, [0]] = pos_data_in[:, [0]] / 1E9

# calculate euler angles from quaternions
euler_angles = np.zeros((ors_data_in.shape[0], 1))

for i in range(ors_data_in.shape[0]):
    ori_quat = ors_data_in[i, [1, 2, 3, 4]]
    ori_rot = R.Rotation.from_quat(ori_quat)
    ori_euler = ori_rot.as_euler("zyx", degrees=False)
    euler_angles[i,0] = ori_euler[0]

# unbound orientation data
ors_euler = np.zeros((ors_data_in.shape[0], 2))
ors_euler[:,0] = ors_data_in[:, 0]
ors_euler[:,1] = np.unwrap(euler_angles[:,0])

# get t_episode
t_episode = math.floor(max(ors_data_in[:, [0]]))

# interpolate orientation at time steps 0.05s
f = interpolate.interp1d(ors_euler[:,0], ors_euler[:,1], kind='linear')
time_scale = (np.arange(0,t_episode,0.05))
ors_itp = f(time_scale)

# bound orientation between 0 and 2pi
ors_itp_2pi_bounded = [normalize_angle_positive(x) for x in ors_itp]

# interpolate position data at time steps 0.05s
f = interpolate.interp1d(pos_data_in[:,0], pos_data_in[:,1], kind='linear')
x_pos_itp = f(time_scale)

f = interpolate.interp1d(pos_data_in[:,0], pos_data_in[:,2], kind='linear')
y_pos_itp = f(time_scale)

nr_measuremnts = len(time_scale)

# Build data file that can be read out by controller.py to run the simulation
thetas.append(0)
for i in range(nr_measuremnts-1):
    #theta = angleDist(ors_itp[i],ors_itp[i+1] )
    theta = angleDist(ors_itp_2pi_bounded[i],ors_itp_2pi_bounded[i+1] )
    thetas.append(theta)

for i in range(nr_measuremnts):
    agentState['Position'] = (x_pos_itp[i],y_pos_itp[i])
    agentState['Orientation'] = ors_itp_2pi_bounded[i]
    agentStates.append(agentState.copy())


out = (thetas,agentStates, t_episode, dt_robot)

plt.plot(x_pos_itp, y_pos_itp)
#tikzplotlib.save('data/plots/sim/plt_rw_'+rw_run+'_x_y_pos.tex')
plt.show()

# For plotting angular velocities
avs = thetas
avs = [x / dt_robot for x in thetas]
plt.plot(time_scale, avs)
plt.show()

plt.plot(time_scale, ors_itp)
plt.show()

plt.plot(time_scale, ors_itp_2pi_bounded)
plt.show()

plt.plot(time_scale, thetas)
plt.show()

with open(outfile, 'wb') as fl:
    fl.write(pickle.dumps(out))
    fl.close()


