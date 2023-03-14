import numpy as np
import copy

from hdcTargetPeak import targetPeakHD

from helper import angleDistAbs
from neuron import phi, phi_inv

# connectivityFunction(i, j) = weight between i and j
def addHDCAttractor(networkTopology, n, connectivityFunction):
    networkTopology.addLayer('hdc_attractor', n)
    networkTopology.connectLayers('hdc_attractor', 'hdc_attractor', connectivityFunction)
    networkTopology.vectorizeConnections('hdc_attractor', 'hdc_attractor')
def addHDCShiftLayers(networkTopology, n, connHDCL, connLHDC, connHDCR, connRHDC):
    networkTopology.addLayer('hdc_shift_left', n)
    networkTopology.addLayer('hdc_shift_right', n)
    networkTopology.connectLayers('hdc_attractor', 'hdc_shift_left', connLHDC)
    networkTopology.connectLayers('hdc_shift_left', 'hdc_attractor', connHDCL)
    networkTopology.connectLayers('hdc_attractor', 'hdc_shift_right', connRHDC)
    networkTopology.connectLayers('hdc_shift_right', 'hdc_attractor', connHDCR)
    networkTopology.vectorizeConnections('hdc_attractor', 'hdc_shift_left')
    networkTopology.vectorizeConnections('hdc_shift_left', 'hdc_attractor')
    networkTopology.vectorizeConnections('hdc_attractor', 'hdc_shift_right')
    networkTopology.vectorizeConnections('hdc_shift_right', 'hdc_attractor')

def addConj(networkTopology, n, connECDtoCONJ, connHDCtoCONJ):
    networkTopology.addLayer('Conj', (n*n))
    networkTopology.connectLayers('Conj','ecd_ring', connECDtoCONJ)
    networkTopology.connectLayers('Conj','hdc_attractor', connHDCtoCONJ)
    networkTopology.vectorizeConnections('ecd_ring','Conj')
    networkTopology.vectorizeConnections('hdc_attractor','Conj')

def addECDRing(networkTopology, n, connECDHDC):
    networkTopology.addLayer('ecd_ring', n)
    networkTopology.connectLayers('hdc_attractor','ecd_ring', connECDHDC)
    networkTopology.vectorizeConnections('ecd_ring','hdc_attractor')

def addACD(networkTopology, n, connCONJtoACD):
    networkTopology.addLayer('acd_ring', n)
    networkTopology.connectLayers('acd_ring','Conj', connCONJtoACD)
    networkTopology.vectorizeConnections('Conj','acd_ring')

def addConj2(networkTopology, n, connECDtoCONJ, connACDtoCONJ, connCONJ2toHDC):
    networkTopology.addLayer('Conj2', (n*n))
    networkTopology.connectLayers('Conj2','ecd_ring', connECDtoCONJ)
    networkTopology.connectLayers('Conj2','acd_ring', connACDtoCONJ)
    networkTopology.connectLayers('hdc_attractor','Conj2', connCONJ2toHDC)
    networkTopology.vectorizeConnections('ecd_ring','Conj2')
    networkTopology.vectorizeConnections('acd_ring','Conj2')
    networkTopology.vectorizeConnections('Conj2','hdc_attractor')

def addHDC2(networkTopology, n, connCONJ2toHDC, connectionHDC2toHDC):
    networkTopology.addLayer('hdc_ring_2', n)
    networkTopology.connectLayers('hdc_ring_2', 'Conj2', connCONJ2toHDC)
    networkTopology.vectorizeConnections('Conj2', 'hdc_ring_2')

# compute the target firing rates at the neurons' preferred directions
def calcTgtPeakCurrs(networkInstance,layername,center,scale):
    n = len(networkInstance.getLayer(layername))
    F = [targetPeakHD(angleDistAbs(x * (2 * np.pi / n), center)) for x in range(n)]
    return [scale * phi_inv(x) for x in F]

# Injects currents corresponding to the targetPeakHD into a certain layer at position "center"
def setPeak(networkInstance,layername, center,scale=1.0, debug=False):
    def printDebug(s):
        if debug:
            print(s)
    U = calcTgtPeakCurrs(networkInstance,layername,center,scale)
    # apply stimulus
    networkInstance.setStimulus(layername, lambda i: (U[i]))


# initializes HDC by applying current corresponding to firing rates 10% of the target function
def simulateLayer(networkInstance, layername, center=None, currs_stim=None, debug=False):
    def printDebug(s):
        if debug:
            print(s)
    # compute the target firing rates at the neurons' preferred directions
    n = len(networkInstance.getLayer(layername))
    if currs_stim is None:
        currs_stim = [0] * n
    if center is None:
        U = [0] * n
    else:
        F = [targetPeakHD(angleDistAbs(x * (2*np.pi / n), center)) for x in range(n)]
        # compute currents corresponding to 10% of those rates
        U = [phi_inv(x * 0.5) for x in F]

    # timestep: 0.5 ms
    dt = 0.0005
    # stimulus time
    t_stim = 0.05
    # settling time
    t_settle = 0.05
    # interval time
    t_interval = 0.05
    # stopping condition: total change in t_interval less than eps
    eps = 0.1

    # apply stimulus
    networkInstance.setStimulus(layername, lambda i : U[i]+currs_stim[i])

    # simulate for ts timesteps
    printDebug(layername+ " : initialization stimulus applied")
    for i in np.arange(0.0, t_stim, dt):
        networkInstance.step(dt)
    printDebug(layername+" : initialization stimulus removed")

    # remove stimulus
    networkInstance.setStimulus(layername, lambda i : currs_stim[i])

    # simulate for t_settle
    for i in np.arange(0.0, t_settle, dt):
        networkInstance.step(dt)

    # simulate in episodes of ts timesteps until the total change in an episode is less than eps
    delta = 2*eps
    it = 0
    while delta > eps:
        it += 1
        delta = 0
        for _ in np.arange(0.0, t_interval, dt):
            ratesBefore = copy.copy(networkInstance.getLayer(layername))
            networkInstance.step(dt)
            ratesAfter = networkInstance.getLayer(layername)
            delta += sum([abs(ratesAfter[i] - ratesBefore[i]) for i in range(n)])
        printDebug(layername+" simulation: iteration {} done with total change {}".format(it, delta))
    printDebug(layername+" simulation: done.")