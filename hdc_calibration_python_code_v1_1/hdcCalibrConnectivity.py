import sys
import numpy
import numpy as np
from matplotlib import pyplot as plt
from params import n_hdc, lam_hdc
from scipy.ndimage.interpolation import shift
from neuron import phi, phi_inv, phi_vec
from scipy.optimize import minimize
from scipy.fftpack import fft, ifft
import pickle
import math
from hdcTargetPeak import targetPeakHD, targetPeakCONJ
from helper import angleDistAbs,writeDataToFile
from hdcOptimizeAttractor import getHDCActivity


#numpy.set_printoptions(threshold=sys.maxsize)

# Set scale to adjust the ACDCs' peak firing rate introduced from CONJ
ACDScale = 0.2

# Set the scale for creating the target calibration currents from CONJ2 to the HDCs
calibr_scale = 0.1

# Calculate weights between HD ring (weights for ECD are identical) to CONJ ###
def calcWeightsFourier(RatesFrom, TargetCurrentsTo, lambda_):

    U = [u for u in TargetCurrentsTo]#[0.5*f2 for f2 in TargetCurrentsTo]
    U_ = fft(U)
    F_ = fft(RatesFrom)
    W_ = []
    for i in range(len(F_)):
        W_.append((U_[i] * F_[i]) / (lambda_ + (abs(F_[i])) ** 2))
    w = np.real(ifft(W_))
    return w

def div0( a, b ):
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide( a, b )
        c[ ~ np.isfinite( c )] = math.inf
    return c[0]

def opt_Fourier_Lambda(rates_from, currs_tgt_to,AnalysisValues):
    rates_tgt_to = np.array([phi(f) for f in currs_tgt_to])
    steps = len(AnalysisValues)
    rates_err_sum = np.zeros(steps)

    for x in range(steps):
        w_opt = calcWeightsFourier(rates_from, currs_tgt_to, AnalysisValues[x])
        W_opt = np.zeros((n_hdc, n_hdc))
        for i in range(n_hdc):
            W_opt[i][:] = w_opt
            w_opt = shift(w_opt, 1, cval=w_opt[(n_hdc - 1)])

        currs_calc_opt = np.matmul(rates_from, W_opt)
        rates_calc_opt = phi_vec(currs_calc_opt)
        rates_err_sum[x] = np.sum((np.abs([a_i - b_i for a_i, b_i in zip(rates_calc_opt, rates_tgt_to)]))**2)
        regularize = div0([1], [AnalysisValues[x]])
        if (regularize==math.inf):
            regularize = 2*div0([1], [AnalysisValues[x+1]])
        rates_err_sum[x] += regularize
    lambda_opt = AnalysisValues[np.where(rates_err_sum==np.nanmin(rates_err_sum))[0][0]]
    #print("lambda_opt",lambda_opt)
    return lambda_opt,rates_err_sum

def calcWeightsScipy(RatesFrom, TargetCurrentsTo):
    # Calc weights by solving a linear equation system with (targetRatesMatHD = input, targetCurrentsCONJ= output)
    targetRatesMatHD = np.zeros((n_hdc, n_hdc))
    InRatesHelper = list(RatesFrom)
    for i in range(n_hdc):
        targetRatesMatHD[i][:] = InRatesHelper
        InRatesHelper.append(InRatesHelper[0])
        del InRatesHelper[0]
    n = len(TargetCurrentsTo)
    fun = lambda x: np.linalg.norm(np.dot(targetRatesMatHD, x) - TargetCurrentsTo)
    sol = minimize(fun, np.zeros(n), method='L-BFGS-B', bounds=[(None, None) for x in range(n)])
    return sol['x']

def calcWeights(RatesFrom,TargetCurrsTo, optMethod, lamsAnsys):
    if optMethod=='Scipy':
        return calcWeightsScipy(RatesFrom, TargetCurrsTo),None
    if optMethod=='Fourier':
        lambda_opt, rates_err_sums = opt_Fourier_Lambda(RatesFrom, TargetCurrsTo, lamsAnsys)
        return calcWeightsFourier(RatesFrom, TargetCurrsTo, lambda_opt),rates_err_sums

def calcInstanceValues(w,RatesFrom,TargetCurrentsTo):
    NrofMats = (n_hdc**RatesFrom.ndim)//n_hdc
    RatesFrom = RatesFrom.reshape((n_hdc**RatesFrom.ndim))
    w = w.reshape(n_hdc)
    W_ = np.zeros((n_hdc*NrofMats, n_hdc))
    for i in range(NrofMats*n_hdc):
        W_[i][:] = w
        if ((i+1)%(n_hdc)!=0):
            w = shift(w, 1, cval=w[(n_hdc - 1)])
    calcCurrents = np.matmul(RatesFrom, W_)
    ErrorCurrents = [a_i - b_i for a_i, b_i in zip(calcCurrents, TargetCurrentsTo)]
    calcRates = phi_vec(calcCurrents)
    return calcCurrents, ErrorCurrents, calcRates

def sumCONJFieldsDiagonals(CONJField):
    conjFieldSum = np.zeros(n_hdc)
    for i in range(n_hdc):
        for a in range(n_hdc):
            conjFieldSum[i]+=CONJField[a][(a+i)%n_hdc]
    return conjFieldSum

def createCONJField(OneDimInput):
    conjField = np.zeros((n_hdc, n_hdc))
    for i in range(n_hdc):
        conjField[i][:] = OneDimInput
    conjField += np.transpose(conjField)
    return conjField

def plotOptResults(fig_nr, fromLayerName, toLayerName, ratesFrom, currs_tgt, w, lamsAnsys, err_sums):

    rates_tgt = [phi(i) for i in currs_tgt]
    currs_calc, currs_err, rates_calc = calcInstanceValues(w, ratesFrom, currs_tgt)
    rates_err = rates_tgt-rates_calc
    fig = plt.figure(fig_nr,figsize=(16, 8))
    fig.suptitle('Weights calculation: '+fromLayerName+" to "+toLayerName, fontsize=20)

    ax1 = plt.subplot(231)
    ax1.plot(np.arange(n_hdc), currs_calc, color='blue', label="Calculated Current")
    ax1.plot(np.arange(n_hdc), currs_tgt, color='green', label="Target Current")
    ax1.plot(np.arange(n_hdc), currs_err, color='yellow', label="Error Current")
    ax1.set_xlabel('Neuron Number i')
    ax1.set_ylabel('Input Current')

    ax2 = plt.subplot(232)
    ax2.plot(np.arange(n_hdc), currs_err, color='yellow')
    ax2.set_xlabel('Neuron Number i')
    ax2.set_ylabel('Error Input Current')

    ax3 = plt.subplot(233)
    ax3.plot(np.arange(n_hdc), rates_calc, color='blue', label="Calculated Rate")
    ax3.plot(np.arange(n_hdc), rates_tgt, color='green', label="Target Rate")
    ax3.set_xlabel('Neuron Number i')
    ax3.set_ylabel('Firing Rate in [Hz]')

    ax4 = plt.subplot(234)
    ax4.plot(np.arange(n_hdc), w, color='orange')
    ax4.set_xlabel('Neuron Number i')
    ax4.set_ylabel('Connection Weight')

    if (err_sums is None)==False:
        ax5 = plt.subplot(235)
        ax5.plot(lamsAnsys, err_sums, color='red')
        ax5.set_xlabel('Optimization Parameter Lambda')
        ax5.set_ylabel('Sum of Error Rate in [Hz]')

    ax6 = plt.subplot(236)
    ax6.plot(np.arange(n_hdc), rates_err, color='yellow')
    ax6.set_xlabel('Neuron Number i')
    ax6.set_ylabel('Error Firing Rate in [Hz]')

    ax1.legend()
    ax3.legend()
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.4)
    #fig.tight_layout()
    plt.show()

def CalcAndPlotWeights():
    #Set optimization algorithm either "Scipy" or "Fourier"
    optMethod = "Fourier"
    #optMethod = "Scipy"


    rates_tgt_peak = np.array([targetPeakHD(angleDistAbs(x * (2 * np.pi / n_hdc), 0)) for x in range(n_hdc)])
    rates_tgt_conjunctive = np.array([targetPeakCONJ(angleDistAbs(x * (2 * np.pi / n_hdc), 0)) for x in range(n_hdc)])

    ## 1. Calculate weights ECD/ACD -> CONJ1
    # Set target properties CONJ
    inputWeight = 0.5  #0.5 from HD cells and 0.5 from ECD cells

    # Calculate target currents to CONJ
    currs_tgt_CONJ_hf =  [inputWeight*(phi_inv(i)) for i in rates_tgt_conjunctive]

    lamsAnsys = np.linspace(700, 720, 100) # Set analysis values for fourier optimization
    wECDtoCONJ,rates_err_sums_ECDtoCONJ = calcWeights(rates_tgt_peak, currs_tgt_CONJ_hf, optMethod, lamsAnsys)
    writeDataToFile(wECDtoCONJ, 'data/model/weights/w_ecd_to_conj.obj')
    writeDataToFile(wECDtoCONJ, 'data/model/weights/w_acd_to_conj.obj')

    plotOptResults(1, "ECDxACD", "CONJ", rates_tgt_peak, currs_tgt_CONJ_hf, wECDtoCONJ, lamsAnsys, rates_err_sums_ECDtoCONJ)

    ## 2. Calculate weights CONJ1 -> ACD
    # Set input rates from CONJ1
    rates_tgt_CONJ_diag_sum = sumCONJFieldsDiagonals(phi_vec(createCONJField(currs_tgt_CONJ_hf)))

    # Calculate target currents to ACD
    currs_tgt_ACD = np.array([phi_inv(f)*ACDScale for f in rates_tgt_peak])

    lamsAnsys = np.linspace(0.5, 1.5, 100)
    wCONJtoACD,rates_err_sums_CONJtoACD = calcWeights(rates_tgt_CONJ_diag_sum, currs_tgt_ACD, optMethod, lamsAnsys)
    writeDataToFile(wCONJtoACD, 'data/model/weights/w_conj_to_acd.obj')

    plotOptResults(2, "CONJ", "ACD", rates_tgt_CONJ_diag_sum, currs_tgt_ACD, wCONJtoACD, lamsAnsys, rates_err_sums_CONJtoACD)

    ## 3. Calculate weights CONJ2 -> HD  (Adapted from Amir/Zhang: 10% of HD input current curve to calibrate HD)
    # Set input rates
    rates_tgt_CONJ2_diag_sum = rates_tgt_CONJ_diag_sum

    # Set target properties
    currs_tgt_HD = np.array([phi_inv(i) for i in (rates_tgt_peak)])
    curr_tgt_HD_max = max(currs_tgt_HD) * calibr_scale
    currs_tgt_HD_shift = np.add(currs_tgt_HD, -min(currs_tgt_HD))

    # Set target output currents
    currs_tgt_CONJtoHD = np.multiply(currs_tgt_HD_shift, (curr_tgt_HD_max / max(currs_tgt_HD_shift)))

    lamsAnsys = np.linspace(4, 10, 100)
    wCONJtoHD,rates_err_sums_CONJtoHD = calcWeights(rates_tgt_CONJ2_diag_sum, currs_tgt_CONJtoHD, optMethod, lamsAnsys)
    wCONJtoHD = np.array([i-(sum(wCONJtoHD)/n_hdc) for i in wCONJtoHD])

    writeDataToFile(wCONJtoHD, 'data/model/weights/w_conj_to_hd.obj')

    plotOptResults(3, "CONJ", "HD", rates_tgt_CONJ2_diag_sum, currs_tgt_CONJtoHD, wCONJtoHD, lamsAnsys, rates_err_sums_CONJtoHD)

    ## 4. Calculate weights HD -> CONJ1
    # Set HD rates by simulating the HD attractor when the calibration calculated in step 3 is present
    currs_calc_CONJtoHD, _, _ = calcInstanceValues(wCONJtoHD, rates_tgt_CONJ2_diag_sum, currs_tgt_CONJtoHD)
    rates_real_HD_new = getHDCActivity(n_hdc, lam_hdc,currs_calc_CONJtoHD)

    # Set target output currents (reuse from step 1)
    currs_tgt_CONJ_hf = currs_tgt_CONJ_hf

    # Calculate weights from HD to CONJ
    lamsAnsys = np.linspace(640, 660, 100)
    wHDtoCONJ,rates_err_sums_HDtoCONJ = calcWeights(rates_real_HD_new, currs_tgt_CONJ_hf, optMethod, lamsAnsys)
    writeDataToFile(wHDtoCONJ, 'data/model/weights/w_hd_to_conj.obj')

    plotOptResults(4, "HD", "CONJ", rates_real_HD_new, currs_tgt_CONJ_hf, wHDtoCONJ, lamsAnsys, rates_err_sums_HDtoCONJ)

if __name__ == "__main__":
    CalcAndPlotWeights()
