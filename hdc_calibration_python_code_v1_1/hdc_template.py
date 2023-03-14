from network import NetworkTopology
import hdcNetwork
import numpy as np
from hdcAttractorConnectivity import HDCAttractorConnectivity
from params import n_hdc, lam_hdc
from scipy.optimize import curve_fit
from helper import loadDataFromFile,writeDataToFile


def norm_shift(i, j, n, amp, sigma):
    if abs(i - j) < float(n) / 2.0:
        # shortest path between i and j doesn't pass 0
        dist = i - j
    else:
        # shortest path between i and j passes 0
        dist = i - (j - n)
        if i > j:
            dist = (i - n) - j
        elif i < j:
            dist = i + (n - j)
    x = (dist / float(n)) * 2 * np.pi
    s = 2*(sigma**2)
    val = amp * (np.sqrt(2*np.e) / np.sqrt(s))*x*np.exp(-(1/s)*(x**2))
    return val

# generates hdc network instance with parameters as defined in the thesis
def generateHDC(useFittingFunction=False, debug=False, InitHD=0.0, place_enc_fdbk=False, simpleFdbk=False):

    # generate connectivity instance (includes hd<->hd weights calculating)
    attrConn = HDCAttractorConnectivity(n_hdc, lam_hdc)
    # load calibration loop weights from files
    attrConn.InitCalibrLoopWeights()
    # generate connectivity functions
    connHDCtoHDC = attrConn.connectionHDCtoHDC  # connectivity HDC -> HDC
    connHDCtoSHIFT = lambda i, j: connHDCtoHDC(i, j) * 0.5  # conn HDC -> SHIFT
    if simpleFdbk == True: connECDtoHDC = attrConn.connectionECDtoHDC  # connectivity ECD -> HDC
    else: connECDtoHDC = lambda i, j: 0
    connECDtoCONJ = attrConn.connectionECDtoCONJ  # connectivity ECD -> CONJ
    connHDCtoCONJ = attrConn.connectionHDCtoCONJ  # connectivity HDC -> CONJ
    connCONJtoACD = attrConn.connectionCONJtoACD  # connectivity CONJ -> ACD
    connECDtoCONJ2 = attrConn.connectionECDtoCONJ2  # connectivity ECD -> CONJ2
    connACDtoCONJ2 = attrConn.connectionACDtoCONJ2  # connectivity ACD -> CONJ
    if place_enc_fdbk==False: connCONJ2toHDC = lambda i, j: 0 # disconnect CONJ2 -> HDC
    else: connCONJ2toHDC = attrConn.connectionCONJ2toHDC  # connectivity CONJ2 -> HDC
    connHDC2toHDC = attrConn.connectionHDC2toHDC  # connectivity HDC2 -> HDC
    if useFittingFunction:
        def fittingFunction(x, A, B, C):
            return A*np.exp(-B*(x**2)) + C
        numDists = int(n_hdc / 2)
        maxDist = np.pi
        X = np.linspace(0.0, maxDist, numDists)
        Y = attrConn.w_hd_to_hd[0:numDists]
        popt, _ = curve_fit(fittingFunction, X, Y)
        # hack: write back to attrConn
        attrConn.w_hd_to_hd = [fittingFunction(x, *popt) for x in X]
        connHDCtoHDC = attrConn.connectionHDCtoHDC
        '''
        if debug:
            plt.plot(X, [fittingFunction(x, *popt) for x in X], label="Fitting function $Ae^{-Bx^2}+C$")
            plt.plot(X, Y, label="Connectivity from [Zha96]")
            plt.legend()
            plt.show()
        '''
    # initialize hdc
    topo = NetworkTopology()
    # two helper functions for SHIFT -> HDC Connection
    offset = 5  # offset in neurons
    strength = 1.0
    def peak_right(to, fr):
        return strength * connHDCtoHDC(to, (fr + offset) % n_hdc)
    def peak_left(to, fr):
        return strength * connHDCtoHDC(to, (fr - offset) % n_hdc)
    # add neural layers to the topology
    hdcNetwork.addHDCAttractor(topo, n_hdc, connHDCtoHDC)
    hdcNetwork.addHDCShiftLayers(topo, n_hdc, connHDCtoSHIFT, lambda i, j : peak_right(i, j) - peak_left(i, j), connHDCtoSHIFT, lambda i, j : peak_left(i, j) - peak_right(i, j))
    hdcNetwork.addECDRing(topo, n_hdc, connECDtoHDC)
    hdcNetwork.addConj(topo, n_hdc, connECDtoCONJ, connHDCtoCONJ)
    hdcNetwork.addACD(topo, n_hdc, connCONJtoACD)
    hdcNetwork.addConj2(topo, n_hdc, connECDtoCONJ2, connACDtoCONJ2, connCONJ2toHDC)
    # TEST: Inject calibration feedback signal into an additional neuron ring (not affecting HD calibration)
    hdcNetwork.addHDC2(topo, n_hdc, connCONJ2toHDC, connHDC2toHDC)
    # Store created topology file if run_topo_from_data = False (Saves the last created topology with
    # either calibration feedback "on" or "off")

    # make instance
    hdc = topo.makeInstance()

    # initialize
    hdcNetwork.simulateLayer(hdc, 'hdc_attractor', InitHD)

    return hdc