from pybullet_environment import PybulletEnvironment
from tqdm import tqdm
import numpy as np
import pickle

# simulation environment, available models: "maze", "plus", "box"
env_model = "box"
# total episode time in seconds
t_episode = 60 #375
# output file
outfile = "data/sim/sim_data_"+str(env_model)+"_"+str(t_episode)+"sec.p"
# robot timestep
dt_robot = 0.05
# the simulation environment window can be turned off, speeds up the simulation significantly
env_visualize = False
# visual cue position (not used, only necessary for initialization)
vis_cue_pos = [0, 0, 2]

env = PybulletEnvironment(1/dt_robot, env_visualize, env_model,vis_cue_pos)
env.reset()

thetas = []
agentStates = []

for t in tqdm(np.arange(0.0, t_episode, dt_robot)):
    # Get orientation difference theta and the agentState from the PyBullet Sim
    theta,agentState = env.step([])
    thetas.append(theta)
    agentStates.append(agentState.copy())

# Write the information into a tuple and save it as pickle file folder data/sim/
out = (thetas,agentStates, t_episode, dt_robot)

with open(outfile, 'wb') as fl:
    fl.write(pickle.dumps(out))
    fl.close()