import matplotlib.pyplot as plt
import numpy as np

import hdcNetwork
from network import NetworkTopology
from hdcAttractorConnectivity import HDCAttractorConnectivity
from hdcTargetPeak import targetPeakHD
from helper import centerAnglesWithY, radToDeg
from tqdm import tqdm
from params import lam_hdc, n_hdc

import time

# returns the stable acitivity peak of a desired layer with desired external stimulus
def getHDCActivity(n, lam, currs_stim=None):
    connectivity = HDCAttractorConnectivity(n, lam)
    topo = NetworkTopology()
    hdcNetwork.addHDCAttractor(topo, n, connectivity.connectionHDCtoHDC)
    inst = topo.makeInstance()
    hdcNetwork.simulateLayer(inst,'hdc_attractor', 0.0, currs_stim)
    result = inst.getLayer('hdc_attractor')
    del topo
    return result


# plot activity peaks for all lambda values in lams
def plotActivityPeaks(n=n_hdc, lams=[500, 21000, 35000]):
    # generate
    X = np.linspace(0.0, 2*np.pi * ((n-1) / n), n)
    target = [targetPeakHD(x) for x in X]
    results = []
    for lam in lams:
        results.append(getHDCActivity(n, lam))
    # plot 
    A, B = centerAnglesWithY(X, target)
    plt.plot(radToDeg(A), B, label="target activity peak")
    plt.xlabel("preferred direction (deg)")
    plt.ylabel("firing rate (Hz)")
    for i in range(len(results)):
        A, B = centerAnglesWithY(X, results[i])
        plt.plot(radToDeg(A), B, label="simulation, $\lambda={}$".format(lams[i]))
    plt.legend()
    plt.show()

# plot sum of errors between target firing rate and actual firing rate for different values of lambda
def plotErrorLambda(lams, logarithmic, n=n_hdc):
    # generate
    X = np.linspace(0.0, 2*np.pi * ((n-1) / n), n)
    target = [targetPeakHD(x) for x in X]
    errs = []
    for lam in tqdm(lams):
        result = getHDCActivity(n, lam)
        # compute total error
        err = sum([abs(target[i] - result[i]) for i in range(n)])
        errs.append(err)
    plt.xlabel("lambda")
    plt.ylabel("sum of errors")
    minerr = errs[0]
    bestlam = lams[0]
    for i in range(len(errs)):
        err = errs[i]
        if err < minerr:
            minerr = err
            bestlam = lams[i]
    print("minimum error: {}, best lambda: {}".format(min(errs), bestlam))
    if logarithmic:
        plt.xscale("log")
    plt.plot(lams, errs)
    plt.show()

# plot the weigth function for one specific lambda
def plotWeightFunction(n=n_hdc, lam=lam_hdc):
    connectivity = HDCAttractorConnectivity(n, lam)
    X = range(50)
    Y = [connectivity.connectionHDCtoHDC(0, x) for x in X]
    plt.xlabel("distance in intervals between neurons")
    plt.ylabel("synaptic weight")
    plt.hlines(0.0, 0.0, 49.0, colors="k", linestyles="--", linewidth=1)
    plt.plot(X, Y)
    plt.show()


# plot the activity peak, the target activity peak and the error between them for one specific lambda
def plotSinglePeak(n=n_hdc, lam=lam_hdc):
    # generate
    X = np.linspace(0.0, 2*np.pi * ((n-1) / n), n)
    target = [targetPeakHD(x) for x in X]
    result = getHDCActivity(n, lam)
    # plot 
    A, B = centerAnglesWithY(X, target)
    plt.plot(radToDeg(A), B, label="target activity peak")
    plt.xlabel("preferred direction (deg)")
    plt.ylabel("firing rate (Hz)")
    C, D = centerAnglesWithY(X, result)
    plt.plot(radToDeg(C), D, label="simulation activity peak")
    plt.plot(radToDeg(A), [abs(b - d) for (b, d) in zip(B, D)], label="error")
    print("Maximum error: {}".format(max([abs(b - d) for (b, d) in zip(B, D)])))
    print("Peak firing rate: {}".format(max(D)))
    print("Target peak firing rate: {}".format(max(B)))
    plt.legend()
    plt.show()