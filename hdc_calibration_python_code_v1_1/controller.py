from pybullet_environment import PybulletEnvironment
import time
import numpy as np
import math
from tqdm import tqdm
import hdcNetwork
from hdcAttractorConnectivity import HDCAttractorConnectivity
from network import NetworkTopology
import matplotlib.pyplot as plt
from polarPlotter import PolarPlotter
import helper
import placeEncoding
from params import n_hdc, weight_av_stim
from hdc_template import generateHDC
from scipy.stats import pearsonr
import random
import pickle
import sys
from hdcCalibrConnectivity import ACDScale


###             set parameters                          ###
##  set environment (pybullet) parameter                ##
#   run pybullet simulation online or from folder data/sim/
run_sim_from_data = False

if run_sim_from_data == True:
    # path where the simulation data is stored (Folder data/sim/)
    sim_data_file = "data/sim/sim_data_box_10sec.p"
    # sim_data_file is a pickle file containing a tuple (thetas, agentStates, t_episode, dt_robot)
    # thetas: array of the change in angle during every time step
    # agentStates: array of agents' positions and orientations at every time step
    if sim_data_file.find('maze') > -1: env_model = "maze"
    elif sim_data_file.find('plus') > -1: env_model = "plus"
    elif sim_data_file.find('box') > -1: env_model = "box"
    elif sim_data_file.find('rw') > -1:
        if sim_data_file.find('circle') > -1: env_model = "rw_circle"
        elif sim_data_file.find('cross') > -1: env_model = "rw_cross"
        elif sim_data_file.find('loops') > -1: env_model = "rw_loops"


elif run_sim_from_data == False:
    # total episode time in seconds
    t_episode = 25
    # in box environment: first calibration until 2.9s, then PI until 10.5s, then calibration until 13.5s
    # robot timestep
    dt_robot = 0.05
    # simulation environment, available models: "maze", "plus", "box"
    env_model = "box"
    # the simulation environment window can be turned off, speeds up the simulation significantly
    env_visualize = True

# set visual cue position in the environment depending on the used environment
if env_model=="maze": vis_cue_pos = [8.5, -7.7, 0]
elif env_model == "plus": vis_cue_pos = [0, -2.7, 0]
elif env_model == "box": vis_cue_pos = [1.05, 0, 0]
elif env_model == "rw_circle": vis_cue_pos = [1850, 650, 0] #rwrun 1
elif env_model == "rw_loops": vis_cue_pos = [1350, 400, 0] #rw run 2
elif env_model == "rw_cross": vis_cue_pos = [1250,500,0] #rw run 3


##  set HD model parameter                              ##

# minimum neuron model timestep
dt_neuron_min = 0.0005 # (2000Hz)

# Choose calibration model (place encoding or simple feedback model) by setting it to "True"
# If no calibration model is chosen (both = False), the HD is only estimated on path integration
# Attention: If both options are set to True, both calibration models are mixed
simple_fdbk = False
place_enc_fdbk = True

# Set calibration mode for the place encoding feedback model
# FGL "on" True: Stores only the ACD, cue distance and the agent's position at the first glance at the cue/landmark
# to reset HD
# FGL "off" False: Associate ACDs in every newly discovered position, to reset HD when revisiting these positions.
FirstGlanceLRN = True
viewAngle = (np.pi)/2

# Attention: The visualization of the vectors is very slow for large matrix dimensions.
# set place encoding parameters
matrixRealLengthCovered = 4 # size of space in the environment that is encoded by the position encoder
matrixDim = 3*matrixRealLengthCovered # place encoding granularity

##  matplotlib visualization/plotting                   ##
rtplot = True
plotfps = 1.0
###             END: set parameters                     ###

####### noisy angular velocity ########
# The HDC network wasn't found to be less sensitive to any type of noise, thus noise isn't included in interactive parameter selection.
use_noisy_av = False

# gaussian noise
# create noise for angular velocity input (used to test PI process, but not used to test HD calibration)
# relative standard deviation (standard deviation = rel. sd * av)
noisy_av_rel_sd = 0.0
# absolute standard deviation (deg)
noisy_av_abs_sd = 0.0

# noise spikes
# average noise spike frequency in Hz
noisy_av_spike_frequency = 0.0
# average magnitude in deg/s
noisy_av_spike_magnitude = 0.0
# standard deviation in deg/s
noisy_av_spike_sd = 0.0

# noise oscillation
noisy_av_osc_frequency = 0.0
noisy_av_osc_magnitude = 0.0
noisy_av_osc_phase = 0.0
#######################################

# Initialize environment
if not run_sim_from_data:
    env = PybulletEnvironment(1/dt_robot, env_visualize, env_model,vis_cue_pos)
    env.reset()
    realDir = env.euler_angle[2]
else:
    with open(sim_data_file, "rb") as fl:
        (thetas_data, agent_states_data, t_episode, dt_robot) = pickle.load(fl)
        fl.close()
    realDir = agent_states_data[0]['Orientation']

# initialize Position Encoder
PosEncoder = placeEncoding.PositionalCueDirectionEncoder(vis_cue_pos[0:2], matrixDim, matrixRealLengthCovered, FirstGlanceLRN=FirstGlanceLRN)
ACD_PeakActive = False

# find dt_neuron < dt_neuron_min = dt_robot / timesteps_neuron with timesteps_neuron integer
timesteps_neuron = math.ceil(dt_robot/dt_neuron_min)
dt_neuron = dt_robot / timesteps_neuron

print("neuron timesteps: {}, dt={}".format(timesteps_neuron, dt_neuron))

# init plotter
nextplot = time.time()
if rtplot:
    plotter = PolarPlotter(n_hdc, 0.0, False, PosEncoder)

# init HDC
print("-> generate HDC") # takes a few minutes
hdc = generateHDC(InitHD=realDir, place_enc_fdbk=place_enc_fdbk, simpleFdbk=simple_fdbk)
print("-> HDC generated")

r2d = (360 / (2 * np.pi)) # rad to deg factor
avs = []
errs = []
errs_signed = []
thetas = []
netTimes = []
robotTimes = []
plotTimes = []
transferTimes = []
decodeTimes = []

#cue_view_pos = []

errs_noisy_signed = []
noisyDir = 0.0

t_before = time.time()
t_ctr = 0

for t in tqdm(np.arange(0.0, t_episode, dt_robot)):
    def getStimL(ahv):
        if ahv < 0.0:
            return 0.0
        else:
            return ahv * weight_av_stim
    def getStimR(ahv):
        if ahv > 0.0:
            return 0.0
        else:
            return - ahv * weight_av_stim
    def getNoisyTheta(theta):
        noisy_theta = theta
        # gaussian noise
        if noisy_av_rel_sd != 0.0:
            noisy_theta = random.gauss(noisy_theta, noisy_av_rel_sd * theta)
        if noisy_av_abs_sd != 0.0:
            noisy_theta = random.gauss(noisy_theta, noisy_av_abs_sd * dt_robot * (1/r2d))
        # noise spikes
        if noisy_av_spike_frequency != 0.0:
            # simplified, should actually use poisson distribution
            probability = noisy_av_spike_frequency * dt_robot
            if random.random() < probability:
                deviation = random.gauss(noisy_av_spike_magnitude * dt_robot * (1/r2d), noisy_av_spike_sd * dt_robot * (1/r2d))
                print(deviation)
                if random.random() < 0.5:
                    noisy_theta = noisy_theta + deviation
                else:
                    noisy_theta = noisy_theta - deviation
        # noise oscillation
        if noisy_av_osc_magnitude != 0.0:
            noisy_theta += noisy_av_osc_magnitude * dt_robot * (1/r2d) * np.sin(noisy_av_osc_phase + noisy_av_osc_frequency * t)
        return noisy_theta

    # robot simulation step
    action = []
    beforeStep = time.time()
    if not run_sim_from_data:
        theta, agentState = env.step(action)
    else:
        theta = thetas_data[t_ctr]
        agentState = agent_states_data[t_ctr]

    afterStep = time.time()
    robotTimes.append((afterStep - beforeStep))
    thetas.append(theta)

    # current is calculated from angular velocity
    angVelocity = theta * (1.0/dt_robot)
    # add noise
    noisy_theta = getNoisyTheta(theta)
    av_net = noisy_theta * (1.0/dt_robot) if use_noisy_av else angVelocity
    avs.append(angVelocity)
    stimL = getStimL(av_net)
    stimR = getStimR(av_net)


    # Calculate the visual cue's ego- & allocentric direction from agent's position & orientation
    ACDir = helper.calcACDir(agentState['Position'], vis_cue_pos[0:2])
    ECDir = helper.calcECDir(agentState['Orientation'], ACDir)

    # simulate network
    beforeStep = time.time()
    #set stimulus to shift layers
    hdc.setStimulus('hdc_shift_left', lambda _ : stimL)
    hdc.setStimulus('hdc_shift_right', lambda _ : stimR)

    ## Set stimulus to ACD-layer & ECD-layer
    # Check if the visual cue is in the agent's field of vision
    cueInSight = helper.cueInSight(viewAngle, ECDir)

    # Check if the agent is in range for learning or restoring the cue's allocentric cue direction
    cueInRange = PosEncoder.checkRange(agentState['Position'])

    if (cueInRange == True) and (cueInSight == True) :

        # Set ECD stimuli to ECD cells
        hdcNetwork.setPeak(hdc, 'ecd_ring', ECDir)

        # save positions in which the agent perceived the cue (only testing)
        #cue_view_pos.append(agentState['Position'])
        #cue_view_pos.append(t)

        if (ACD_PeakActive == True):
            # decode the ACD encoded by ACD cells
            # ACD_PeakActive takes care that ACD is only derived from a fully emerged activity peak
            decodedACDDir = helper.decodeRingActivity(list(hdc.getLayer('acd_ring')))
            # Set ACD = False if ACD learned in new position
            # Set ACD = restored ACD for calibration
            ACD = PosEncoder.get_set_ACDatPos(agentState['Position'], decodedACDDir)

            if (ACD != False):
                # Set ACD stimuli to ACD cells
                hdcNetwork.setPeak(hdc, 'acd_ring', ACD,scale=(1-ACDScale))
        ACD_PeakActive = True
    else:
        # Set stimuli = 0 when cue is out of sight
        hdc.setStimulus('ecd_ring', lambda i: 0)
        hdc.setStimulus('acd_ring', lambda i: 0)
        ACD_PeakActive = False


    hdc.step(dt_neuron, numsteps=timesteps_neuron)
    afterStep = time.time()
    netTimes.append((afterStep - beforeStep) / timesteps_neuron)

    # Get layer rates
    rates_hdc       = list(hdc.getLayer('hdc_attractor'))
    rates_sl        = list(hdc.getLayer('hdc_shift_left'))
    rates_sr        = list(hdc.getLayer('hdc_shift_right'))
    rates_ecd       = list(hdc.getLayer('ecd_ring'))
    rates_conj      = list(hdc.getLayer('Conj'))
    rates_acd       = list(hdc.getLayer('acd_ring'))
    rates_conj_2    = list(hdc.getLayer('Conj2'))
    rates_hdc2      = list(hdc.getLayer('hdc_ring_2'))

    # Calculate & save errors
    beforeStep = time.time()

    # Decode layer activities
    decodedDir      = helper.decodeRingActivity(rates_hdc)
    decodedECDDir   = helper.decodeRingActivity(rates_ecd)
    decodedACDDir   = helper.decodeRingActivity(rates_acd)

    # Calculate direction
    realDir = (realDir + theta) % (2 * np.pi)
    noisyDir = (noisyDir + noisy_theta) % (2 * np.pi)

    err_noisy_signed_rad = helper.angleDist(realDir, noisyDir)
    errs_noisy_signed.append(r2d * err_noisy_signed_rad)
    err_signed_rad = helper.angleDist(realDir, decodedDir)
    errs_signed.append(r2d * err_signed_rad)
    errs.append(abs(r2d * err_signed_rad))

    afterStep = time.time()
    decodeTimes.append(afterStep - beforeStep)

    # plotting
    if time.time() > nextplot and rtplot:
        nextplot += 1.0 / plotfps
        beforeStep = time.time()
        plotter.plot(rates_hdc, rates_sl, rates_sr, rates_ecd, rates_conj, rates_acd, rates_conj_2, rates_hdc2,
                     PosEncoder, stimL, stimR, realDir, decodedDir)
        afterStep = time.time()
        plotTimes.append((afterStep - beforeStep))
    afterStep = time.time()
    t_ctr += 1

# final calculations
t_total = time.time() - t_before
X = np.arange(0.0, t_episode, dt_robot)
cahv = [avs[i] - avs[i - 1] if i > 0 else avs[0] for i in range(len(avs))]
cerr = [errs_signed[i] - errs_signed[i - 1] if i > 0 else errs_signed[0] for i in range(len(errs_signed))]
corr, _ = pearsonr(cahv, cerr)

# error noisy integration vs. noisy HDC
if use_noisy_av:
    plt.plot(X, errs_noisy_signed, label="Noisy integration")
    plt.plot(X, errs_signed, label="Noisy HDC")
    plt.plot([0.0, t_episode], [0.0, 0.0], linestyle="dotted", color="k")
    plt.xlabel("time (s)")
    plt.ylabel("error (deg)")
    plt.legend()
    plt.show()

# print results
#print("cue_view_pos",cue_view_pos)

print("\n\n\n")
print("############### Begin Simulation results ###############")
# performance tracking
print("Total time (real): {:.2f} s, Total time (simulated): {:.2f} s, simulation speed: {:.2f}*RT".format(t_total, t_episode, t_episode / t_total))
print("Average step time network:  {:.4f} ms; {} it/s possible".format(1000.0 * np.mean(netTimes), 1.0/np.mean(netTimes)))
print("Average step time robot:    {:.4f} ms; {} it/s possible".format(1000.0 * np.mean(robotTimes), 1.0/np.mean(robotTimes)))
if rtplot:
    print("Average step time plotting: {:.4f} ms; {} it/s possible".format(1000.0 * np.mean(plotTimes), 1.0/np.mean(plotTimes)))
time_coverage = 0.0
print("Average time decoding:      {:.4f} ms; {} it/s possible".format(1000.0 * np.mean(decodeTimes), 1.0/np.mean(decodeTimes)))
print("Steps done network:  {}; Time: {:.3f} s; {:.2f}% of total time".format(len(X) * timesteps_neuron, len(X) * timesteps_neuron * np.mean(netTimes), 100 * len(X) * timesteps_neuron * np.mean(netTimes) / t_total))
time_coverage += 100 * len(X) * timesteps_neuron * np.mean(netTimes) / t_total
print("Steps done robot:    {}; Time: {:.3f} s; {:.2f}% of total time".format(len(X), len(X) * np.mean(robotTimes), 100 * len(X) * np.mean(robotTimes) / t_total))
time_coverage += 100 * len(X) * np.mean(robotTimes) / t_total
if rtplot:
    print("Steps done plotting: {}; Time: {:.3f} s; {:.2f}% of total time".format(int(t_episode / plotfps), int(t_episode / plotfps) * np.mean(plotTimes), 100 * int(t_episode / plotfps) * np.mean(plotTimes) / t_total))
    time_coverage += 100 * int(t_episode / plotfps) * np.mean(plotTimes) / t_total
print("Steps done decoding: {}; Time: {:.3f} s; {:.2f}% of total time".format(len(X), len(X) * np.mean(decodeTimes), 100 * len(X) * np.mean(decodeTimes) / t_total))
time_coverage += 100 * len(X) * np.mean(decodeTimes) / t_total
print("Time covered by the listed operations: {:.3f}%".format(time_coverage))
print("maximum angular velocity: {:.4f} deg/s".format(max(avs) * r2d))
print("average angular velocity: {:.4f} deg/s".format(sum([r2d * (x / len(avs)) for x in avs])))
print("median angular velocity:  {:.4f} deg/s".format(np.median(avs)))
print("maximum error: {:.4f} deg".format(max(errs)))
print("average error: {:.4f} deg".format(np.mean(errs)))
print("median error:  {:.4f} deg".format(np.median(errs)))
print("################ End Simulation results ################")
print("\n\n\n")

# close real-time plot
plt.close()
plt.ioff()

# plot error and angular velocity
fig, ax1 = plt.subplots()
# ax1.set_xlim(200, 375)
ax1.set_xlabel("time (s)")
ax1.set_ylabel("error (deg)")
ax1.set_ylim(-13.5, 13.5)
ax1.plot(X, errs_signed, color="tab:blue")
ax1.tick_params(axis="y", labelcolor="tab:blue")
ax2 = ax1.twinx()
ax2.set_ylabel("angular velocity (deg/s)")
ax2.set_ylim(-50, 50)
ax2.plot(X, [x * r2d for x in avs], color="tab:orange")
ax2.tick_params(axis="y", labelcolor="tab:orange")
ax1.plot([0.0, t_episode], [0.0, 0.0], linestyle="dotted", color="k")
plt.show()

# plot only error
plt.xlabel("time (s)")
plt.ylabel("error (deg)")
plt.ylim(-1.6, 1.6)
plt.xlim(0.0, t_episode)
plt.plot(X, errs_signed)
plt.plot([0.0, t_episode], [0.0, 0.0], linestyle="dotted", color="k")
plt.show()

# plot only angular velocity
plt.xlabel("time (s)")
plt.ylabel("angular velocity (deg/s)")
plt.ylim(-50, 50)
plt.xlim(0.0, t_episode)
plt.plot(X, [x * r2d for x in avs])
plt.plot([0.0, t_episode], [0.0, 0.0], linestyle="dotted", color="k")
plt.show()

# plot total rotation
totalMovements = [0.0] * len(thetas)
for i in range(1, len(avs)):
    totalMovements[i] = totalMovements[i-1] + abs(r2d * thetas[i - 1])
plt.plot(X, totalMovements)
plt.xlabel("time (s)")
plt.ylabel("total rotation (deg)")
plt.show()

# plot relative error # only used for PI evaluation
# begin after 20%
begin_relerror = int(0.2 * len(X))
plt.plot(X[begin_relerror:len(X)], [100 * errs[i] / totalMovements[i] for i in range(begin_relerror, len(errs))])
plt.xlabel("time (s)")
plt.ylabel("relative error (%)")
plt.show()

# plot change in angular velocity vs. change in error # only used for PI evaluation
plt.scatter(cahv, cerr)
plt.plot([min(cahv), max(cahv)], [corr * min(cahv), corr * max(cahv)], label="linear approximation with slope {:.2f}".format(corr), color="tab:red")
plt.legend()
plt.xlabel("change in angular velocity (deg/s)")
plt.ylabel("change in error (deg)")
plt.show()
if not run_sim_from_data:
    env.close()