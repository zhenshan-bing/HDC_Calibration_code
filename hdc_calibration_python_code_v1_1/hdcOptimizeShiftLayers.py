import hdcNetwork
from hdcAttractorConnectivity import HDCAttractorConnectivity
from network import NetworkTopology
import matplotlib.pyplot as plt
import numpy as np
from params import n_hdc
import helper
from tqdm import tqdm
from scipy import stats
from hdc_template import generateHDC

def testStimuli(plots=[], dt=0.0005, stims = np.linspace(0.0, 1.0, 101)):
#def testStimuli(plots=[], dt=0.05, stims=np.linspace(0.0, 0.2, 10)):

    # use only one network topology
    hdc_topo = generateHDC().topology
    timesteps = int(0.2 / dt)
    recordingStep = int(0.1 / dt)
    angVelocities = []
    for stim in tqdm(stims):
        # set angles to zero if distance to zero is less than epsilon
        def cleanupAngles(angles, epsilon):
            newAngles = []
            for a in angles:
                if abs(2*np.pi - a) < epsilon:
                    newAngles.append(0.0)
                else:
                    newAngles.append(a)
            return newAngles
        def radToDeg(angles):
            newAngles = [0.0] * len(angles)
            for i in range(len(angles)):
                newAngles[i] = (360.0 / (2*np.pi)) * angles[i]
            return newAngles
        def calculateTotalMovement(decDirections):
            total = 0.0
            for i in range(1, len(decDirections)):
                change = helper.angleDist(decDirections[i-1], decDirections[i])
                total += change
            return total
        
        # make hdc network instance
        hdc = hdc_topo.makeInstance()
        #hdcNetwork.initializeHDC(hdc, 0.0)
        hdcNetwork.simulateLayer(hdc, 'hdc_attractor', 0.0)

        # set stimulus
        hdc.setStimulus('hdc_shift_left', lambda _ : stim)
        decDirectionsFull = []
        decDirections = []

        # simulate for 0.1s without recording
        if recordingStep != 0:
            if not ("dir_over_time" in plots):
                hdc.step(dt, numsteps=(timesteps - recordingStep))
            else:
                for i in range(recordingStep):
                    hdc.step(dt, numsteps=1)
                    rates = hdc.getLayer('hdc_attractor')
                    decDirectionsFull.append(helper.decodeRingActivity(rates))

        # simulate and record
        for i in range(timesteps - recordingStep):
            hdc.step(dt, numsteps=1)
            rates = hdc.getLayer('hdc_attractor')
            decDirections.append(helper.decodeRingActivity(rates))
            decDirectionsFull.append(helper.decodeRingActivity(rates))

        # calculate total movement
        [total_movement] = radToDeg([calculateTotalMovement(cleanupAngles(decDirections, 0.001))])
        angVelocities.append((total_movement) / (dt * (timesteps - recordingStep)))
        # plotting
        if "dir_over_time" in plots:
            plt.plot([dt * s for s in range(timesteps)], radToDeg(cleanupAngles(decDirectionsFull, 0.001)), label="$s_{{left}}={}$".format(stim))
    if "dir_over_time" in plots:
        plt.xlabel("time (s)")
        plt.ylabel("decoded direction (deg)")
        plt.legend()
        plt.show()
    if "av_over_stim" in plots:
        # calculate slope from the line going through (0, 0) and the first data point (stim[1], av[1])
        # stim[1] and not stim[0] is used since stim[0] = 0 in the standard stimulus list
        slope = angVelocities[1] / stims[1]
        weight = 1/slope
        # plot the line
        plt.plot([0.0, stims[-1]], [0.0, slope * stims[-1]], label="linear approximation for low stimuli")
        # print the corrsponding weight
        print("Angular velocity v -> stimulus s function: s = v*{:.6f}".format(weight * (360/(2*np.pi))))
        # plot the decoded angular velocity over all stimuli
        plt.plot(stims, angVelocities)
        plt.xlabel("shift-left stimulus")
        plt.ylabel("decoded angular velocity (deg/s)")
        plt.xlim(0.0, stims[-1])
        plt.ylim(0.0, slope * stims[-1])
        plt.legend()
        plt.show()