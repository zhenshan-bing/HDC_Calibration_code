from hdc_template import generateHDC
import matplotlib.pyplot as plt
from helper import centerAnglesWithY, radToDeg
from params import n_hdc, lam_hdc
from hdcAttractorConnectivity import HDCAttractorConnectivity
import numpy as np
from hdcTargetPeak import targetPeakHD
from neuron import phi, r_m
import hdcOptimizeShiftLayers
import hdcOptimizeAttractor
from os import listdir
import pickle
from SimResult import SimResult
from tqdm import tqdm
from scipy.interpolate import interp1d
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Used only by El-Sewisy 2020 to create thesis plots

basedir_simresults_kitti_raw = "results/kitti_raw"

def plotHDCSL():
    hdc = generateHDC()
    rates_shiftl = hdc.getLayer('hdc_shift_left')
    rates_shiftr = hdc.getLayer('hdc_shift_right')
    rates_hdc = hdc.getLayer('hdc_attractor')

    X = np.linspace(0.0, 2*np.pi * ((n_hdc-1) / n_hdc), n_hdc)
    A, B = centerAnglesWithY(X, rates_shiftl)
    A, C = centerAnglesWithY(X, rates_shiftr)
    A, D = centerAnglesWithY(X, rates_hdc)
    plt.plot(radToDeg(A), D, label='HDC Attractor')
    plt.plot(radToDeg(A), B, label='shift layers')
    plt.legend()
    plt.xlabel('preferred direction (deg)')
    plt.ylabel('firing rate (Hz)')
    plt.show()

def plotWeights():
    attrConn = HDCAttractorConnectivity(n_hdc, lam_hdc)
    X = range(-49, 50)
    w_HDC_HDC = [attrConn.connectionHDCtoHDC(0, j % n_hdc) for j in X]
    w_HDC_SL = [attrConn.connectionHDCtoHDC(0, j % n_hdc) * 0.5 for j in X]

    offset = 5
    strength = 1.0
    def peak_right(i, j):
        return strength * attrConn.connectionHDCtoHDC((i + offset) % n_hdc, j)
    def peak_left(i, j):
        return strength * attrConn.connectionHDCtoHDC((i - offset) % n_hdc, j)
    def conn_right(i, j):
        return peak_right(i, j) - peak_left(i, j)
    def conn_left(i, j):
        return peak_left(i, j) - peak_right(i, j)
    
    w_Sleft_HDC = [conn_left(0, j) for j in X]
    w_Sright_HDC = [conn_right(0, j) for j in X]

    plt.plot(X, w_HDC_HDC, label="HDC -> HDC")
    plt.plot(X, w_HDC_SL, label="HDC -> shift layers")
    plt.plot(X, w_Sleft_HDC, label="shift left -> HDC")
    plt.plot(X, w_Sright_HDC, label="shift right -> HDC")

    plt.legend()
    plt.hlines(0.0, -49.0, 49.0, colors="k", linestyles="--", linewidth=1)
    plt.xlabel("distance in intervals between neurons")
    plt.ylabel("synaptic weight")
    plt.show()

def plotTuningCurves():
    r2d = 360 / (2*np.pi)
    X_neuron = range(100)
    X_ang = [(i * 2 * np.pi) / 100 for i in X_neuron]
    Y_peak = [targetPeakHD(x) for x in X_ang]
    Y_neg90 = [targetPeakHD((x + (0.5 * np.pi)) % (2 * np.pi)) for x in X_ang]
    Y_pos90 = [targetPeakHD((x - (0.25 * np.pi)) % (2 * np.pi)) for x in X_ang]
    Y_neg225 = [targetPeakHD((x + (0.125 * np.pi)) % (2 * np.pi)) for x in X_ang]

    X = X_ang

    X, Y_pos90 = centerAnglesWithY(X_ang, Y_pos90)
    X, Y_neg90 = centerAnglesWithY(X_ang, Y_neg90)
    X, Y_neg225 = centerAnglesWithY(X_ang, Y_neg225)
    X, Y_peak = centerAnglesWithY(X_ang, Y_peak)
    plt.plot(radToDeg(X), Y_pos90, label="tuning curve, preferred direction $45^\circ$", color="g")
    plt.plot(radToDeg(X), Y_neg90, label="tuning curve, preferred direction $-90^\circ$", color="tab:orange")
    plt.plot(radToDeg(X), Y_neg225, label="tuning curve, preferred direction $-22.5^\circ$", color="r")


    plt.scatter([0, 45], [targetPeakHD(0.25 * np.pi)] * 2, color="g")
    plt.scatter([0, -90], [targetPeakHD(0.5 * np.pi)] * 2, color="tab:orange")
    plt.scatter([0, -22.5], [targetPeakHD(0.125 * np.pi)] * 2, color="r")


    plt.plot([0, 45], [targetPeakHD(0.25 * np.pi)] * 2, color="g", linestyle="dotted")
    plt.plot([0, -90], [targetPeakHD(0.5 * np.pi)] * 2, color="orange", linestyle="dotted")
    plt.plot([0, -22.5], [targetPeakHD(0.125 * np.pi)] * 2, color="r", linestyle="dotted")

    plt.plot(radToDeg(X), Y_peak, label="activity peak", color="k", linestyle="--")

    plt.vlines([-90, -22.5, 0, 45], 0, targetPeakHD(0.0), color="k", linestyle="dotted", linewidth=1.0)
    plt.xlim(-100, 60)
    plt.ylim(0.0, 100)
    plt.xticks([-90, -22.5, 0, 45])
    plt.xlabel("direction (deg)")
    plt.ylabel("firing rate (Hz)")

    plt.legend()
    plt.show()

def plotPhi():
    X = np.linspace(-5.0, 12.0, 1000)
    plt.plot(X, [phi(x) for x in X])
    plt.xlim(-5.0, 12.0)
    plt.ylim(0.0, r_m)
    plt.plot([-5.0, 12.0], [phi(0.0), phi(0.0)], linestyle="dotted", color="k")
    plt.plot([0.0, 0.0], [0.0, r_m], linestyle="dotted", color="k")
    plt.xlabel("$x$")
    plt.ylabel("$\phi(x)$")
    plt.show()

def plotMultipleActivityPeaks():
    pass

def plotErr(simResults, legend=True):
    plt.xlabel("time (s)")
    plt.ylabel("error (deg)")
    maxAbsErr = max([max(simResult.errs) for simResult in simResults])
    maxTime = max([max(simResult.times) for simResult in simResults])
    plt.ylim(-1.2 * maxAbsErr, 1.2 * maxAbsErr)
    plt.xlim(0.0, maxTime)
    for simResult in simResults:
        plt.plot(simResult.times, simResult.errs_signed, label=simResult.label)
        plt.plot([0.0, maxTime], [0.0, 0.0], linestyle="dotted", color="k")
    if legend:
        plt.legend()
    plt.show()

def plotDirection(simResults, legend=True):
    r2d = 180.0 / np.pi
    plt.xlabel("time (s)")
    plt.ylabel("orientation (deg)")
    for simResult in simResults:
        if legend:
            plt.plot(simResult.times, [r2d * (x - 2*np.pi if x > np.pi else x) for x in simResult.realDirections[1:]], label="{}, true".format(simResult.label))
            plt.plot(simResult.times, [r2d * (x - 2*np.pi if x > np.pi else x) for x in simResult.decDirections], label="{}, HDC".format(simResult.label))
            plt.plot(simResult.times, [r2d * (x - 2*np.pi if x > np.pi else x) for x in simResult.quad_dirs], label="{}, integration".format(simResult.label))
        else:
            plt.plot(simResult.times, [r2d * (x - 2*np.pi if x > np.pi else x) for x in simResult.realDirections[1:]], label="true")
            plt.plot(simResult.times, [r2d * (x - 2*np.pi if x > np.pi else x) for x in simResult.decDirections], label="HDC")
            plt.plot(simResult.times, [r2d * (x - 2*np.pi if x > np.pi else x) for x in simResult.quad_dirs], label="integration")
    plt.legend()
    plt.show()

def printTable(simResults):
    ranges = [[min(s.times), max(s.times)] for s in simResults]
    error_arrays = []
    for simResult in simResults:
        errors = []
        for i in range(len(simResult.quad_errs)):
            errors.append(simResult.errs[i] - simResult.quad_errs[i])
        error_arrays.append(errors)
    print("\\begin{table}[]")
    print("  \\begin{tabular}{|c|c|c|c|}")
    print("    \\hline")
    print("    time (s) & max error & min error & mean error")
    print("    \\hline")
    for i, errors in enumerate(error_arrays):
        print("    {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\".format(ranges[i][1], max(errors), min(errors), np.mean(errors)))
    print("    \\hline")
    print("  \\end{tabular}")
    print("\\end{table}")

def printTableGroundTruth(simResults):
    ranges = [[min(s.times), max(s.times)] for s in simResults]
    error_arrays = [simResult.errs for simResult in simResults]
    print("\\begin{table}[]")
    print("  \\begin{tabular}{|c|c|c|c|}")
    print("    \\hline")
    print("    time (s) & max error & mean error")
    print("    \\hline")
    for i, errors in enumerate(error_arrays):
        print("    {:.4f} & {:.4f} & {:.4f} \\\\".format(ranges[i][1], max(errors), np.mean(errors)))
    print("    \\hline")
    print("  \\end{tabular}")
    print("\\end{table}")

def plotAngularVelocity(simResults, legend=True):
    plt.xlabel("time (s)")
    plt.ylabel("angular velocity (deg/s)")
    for simResult in simResults:
        plt.plot(simResult.times, simResult.avs[1:], label=simResult.label)
    if legend:
        plt.legend()
    plt.show()

def plotErrWithIntegration(simResults, legend=True):
    plt.xlabel("time (s)")
    plt.ylabel("error (deg)")
    for simResult in simResults:
        if legend:
            plt.plot(simResult.times, simResult.errs_signed, label="{}, HDC".format(simResult.label))
            plt.plot(simResult.times, simResult.quad_errs_signed, label="{}, integration".format(simResult.label))
        else:
            plt.plot(simResult.times, simResult.errs_signed, label="HDC")
            plt.plot(simResult.times, simResult.quad_errs_signed, label="integration")
    plt.legend()
    plt.show()


def plotErrAvg(simResults):
    ranges = [[min(s.times), max(s.times)] for s in simResults]
    res = max(len(s.times) for s in simResults)
    grid = np.linspace(min([x[0] for x in ranges]), max([x[1] for x in ranges]), res)
    # average error
    for simResult in simResults:
        errors = simResult.errs
        plt.plot(simResult.times, errors, color="silver")
    plt.plot([grid[0], grid[-1]], [0.0, 0.0], color="k", linestyle="--")
    def inRange(r, x):
        return x >= r[0] and x <= r[1]
    def avg(l):
        return sum(l) / float(len(l))
    errorFunctions = []
    for simResult in simResults:
        errors = simResult.errs
        errorFunctions.append(interp1d(simResult.times, errors, kind="linear"))
    avgs = []
    for x in grid:
        vals = []
        for i in range(len(simResults)):
            if inRange(ranges[i], x):
                vals.append(errorFunctions[i](x))
        avgs.append(avg(vals))
    plt.xlabel("time (s)")
    plt.ylabel("absolute error HDC network vs. ground truth (deg)\n(averaged over all available scenarios)")
    plt.plot(grid, avgs)
    plt.show()


def plotErrVsIntegrationAvg(simResults):
    ranges = [[min(s.times), max(s.times)] for s in simResults]
    res = max(len(s.times) for s in simResults)
    grid = np.linspace(min([x[0] for x in ranges]), max([x[1] for x in ranges]), res)
    # average error
    for simResult in simResults:
        errors = []
        for i in range(len(simResult.quad_errs)):
            errors.append(simResult.errs[i] - simResult.quad_errs[i])
        plt.plot(simResult.times, errors, color="silver")
    plt.plot([grid[0], grid[-1]], [0.0, 0.0], color="k", linestyle="--")
    def inRange(r, x):
        return x >= r[0] and x <= r[1]
    def avg(l):
        return sum(l) / float(len(l))
    errorFunctions = []
    for simResult in simResults:
        errors = []
        for i in range(len(simResult.quad_errs)):
            errors.append(simResult.errs[i] - simResult.quad_errs[i])
        errorFunctions.append(interp1d(simResult.times, errors, kind="linear"))
    avgs = []
    for x in grid:
        vals = []
        for i in range(len(simResults)):
            if inRange(ranges[i], x):
                vals.append(errorFunctions[i](x))
        avgs.append(avg(vals))
    plt.xlabel("time (s)")
    plt.ylabel("absolute error HDC network vs. integration (deg)\n(averaged over all available scenarios)")
    plt.plot(grid, avgs)
    plt.show()

def plotErrVsIntegrationHist(simResults):
    all_errs = []
    for simResult in simResults:
        errors = []
        for i in range(len(simResult.quad_errs)):
            errors.append(simResult.errs[i] - simResult.quad_errs[i])
        all_errs.extend(errors)
    # bucket size in deg
    bucketSize = 0.1
    minErr = min(all_errs)
    maxErr = max(all_errs)
    int_size = maxErr - minErr
    numBuckets = int(np.ceil(int_size / bucketSize) + 1 if (int_size / bucketSize).is_integer else np.ceil(int_size / bucketSize))
    hist = np.zeros((numBuckets), dtype=np.int16)
    for e in np.array(all_errs):
        bucket = int(np.floor((e - minErr) / bucketSize))
        hist[bucket] += 1
    plt.plot(np.linspace(minErr, maxErr, numBuckets), hist)
    plt.plot([minErr, maxErr], [0.0, 0.0], linestyle="dotted", color="k")
    plt.plot([0.0, 0.0], [0.0, max(hist)], linestyle="dotted", color="k")
    avgErr = np.mean(all_errs)
    medianErr = np.median(all_errs)
    plt.plot([avgErr, avgErr], [0.0, max(hist)], linestyle="--", label="mean")
    plt.plot([medianErr, medianErr], [0.0, max(hist)], linestyle="--", label="median")
    plt.xlabel("absolute error HDC network vs. integration (deg)\nNegative error: better than integration")
    plt.ylabel("Number of occurences in buckets of {} deg".format(round(bucketSize, 6)))
    plt.legend()
    plt.show()

def plotErrVsIntegration(simResults, legend=True, printTable=True):
    ranges = [[min(s.times), max(s.times)] for s in simResults]
    res = max(len(s.times) for s in simResults)
    grid = np.linspace(min([x[0] for x in ranges]), max([x[1] for x in ranges]), res)

    plt.xlabel("time (s)")
    plt.ylabel("absolute error HDC network vs. integration (deg)\nNegative error: better than integration")
    all_errs = []
    error_arrays = []
    for simResult in simResults:
        errors = []
        for i in range(len(simResult.quad_errs)):
            errors.append(simResult.errs[i] - simResult.quad_errs[i])
        plt.plot(simResult.times, errors, label=simResult.label)
        all_errs.extend(errors)
        error_arrays.append(errors)
    if legend:
        plt.legend()

    plt.plot([grid[0], grid[-1]], [0.0, 0.0], color="k", linestyle="--")
    plt.show()

def makeXSpace( n, bias, invert):
    return [(x + bias) * (-1 if invert else 1) % (np.pi * 2) for x in np.linspace(0.0, np.pi * 2, (n+1))]

if __name__ == "__main__":
    print("Available Figures:")
    print("3.1:  Function phi")
    print("3.2:  Tuning curves vs. activity profile")
    print("3.4:  Activity peaks for differend values for lambda")
    print("3.5:  Error for different values for lambda (logarithmic)")
    print("3.6:  Error for different values for lambda (linear)")
    print("3.7:  Activity peak vs. target activity peak")
    print("3.8:  Weights inside the attractor network")
    print("3.9:  Activity peaks HDC and shift layers")
    print("3.10: All weights")
    print("3.11: Decoded direction over time during shifting")
    print("3.12: Angular velocities reached with different stimuli")
    print("E1:  Plot figures from datasets using the output file from controller_kitti_raw.py")
    print("------------------ NOTES ------------------")
    print("Figure 3.3 is made directly in LaTEX and can't be plotted here.")
    print("Figures 4.2-4.5 are plotted with controller.py")
    print("Figures 4.6 and 4.7 are plotted with 'E1'")
    print("-------------------------------------------")
    terminate = False
    while not terminate:
        print("Enter figure number (e.g. '3.1') to plot it or close with 'exit': ", end="")
        fig_name = input()
        if fig_name == "3.1":
            plotPhi()
        elif fig_name == "3.2":
            plotTuningCurves()
        elif fig_name == "3.4":
            hdcOptimizeAttractor.plotActivityPeaks()
        elif fig_name == "3.5":
            print("The figure in the thesis is plotted with 10000 lambda values logarithmically distributed from 10^-10 to 10^10. This may take several hours.")
            print("Press Enter to plot the same as in the thesis or enter custom parameters in the form 'f t r' to plot with r values from 10^f to 10^t: ", end="")
            params = input().split()
            if params == []:
                params = ["-10", "10", "10000"]
            params = list(map(int, params))
            hdcOptimizeAttractor.plotErrorLambda(np.logspace(params[0], params[1], params[2]), True)
        elif fig_name == "3.6":
            print("The figure in the thesis is plotted with 10000 lambda values linearly distributed from 25000 to 26000. This may take several hours.")
            print("Press Enter to plot the same as in the thesis or enter custom parameters in the form 'f t r' to plot with r values from f to t: ", end="")
            params = input().split()
            if params == []:
                params = ["25000", "26000", "10000"]
            params = list(map(int, params))
            hdcOptimizeAttractor.plotErrorLambda(np.linspace(params[0], params[1], params[2]), False)
        elif fig_name == "3.7":
            hdcOptimizeAttractor.plotSinglePeak()
        elif fig_name == "3.8":
            hdcOptimizeAttractor.plotWeightFunction()
        elif fig_name == "3.9":
            plotHDCSL()
        elif fig_name == "3.10":
            plotWeights()
        elif fig_name == "3.11":
            hdcOptimizeShiftLayers.testStimuli(plots=["dir_over_time"], stims=[0.0, 0.01, 0.1, 0.5])
        elif fig_name == "3.12":
            hdcOptimizeShiftLayers.testStimuli(plots=["av_over_stim"])
        elif fig_name == "exit":
            terminate = True
        elif fig_name == "E1":
            files = listdir(basedir_simresults_kitti_raw)
            if len(files) == 0:
                print("No files available, generate some with controller_kitti_raw.py")
            elif len(files) == 1:
                print("using file {}".format(files[0]))
                infile = open("{}/{}".format(basedir_simresults_kitti_raw, files[0]), "rb")
            else:
                print("Available files:")
                for i, filename in enumerate(files):
                    print("{}: {}".format(i, filename))
                print("Select file number: ", end="")
                infile = open("{}/{}".format(basedir_simresults_kitti_raw, files[int(input())]), "rb")
            simResults = pickle.load(infile)
            # filter simResults with no motion.
            simResults = list(filter(lambda simResult : simResult.times[-1] >= 10.0 and max(simResult.avs) > 0.5, simResults))
            print("The file contains {} scenarios".format(len(simResults)))
            print("Select one scenario only? (Y/N): ", end="")
            scenarioNum = -1
            if input() == "Y":
                print("Available scenarios:")
                for i, simResult in enumerate(simResults):
                    print("{}: {}, duration: {} s".format(i, simResult.label, simResult.times[-1] - simResult.times[0]))
                print("Select scenario number: ", end="")
                scenarioNum = int(input())
                simResults = [simResults[scenarioNum]]
            terminateInner = False
            while not terminateInner:
                print("Available plots:")
                print("0: Error vs. Integration")
                print("1: Orientation")
                print("2: Angular velocity")
                print("3: HDC Error + Integration Error")
                print("4: Error vs. Integration (Histogram)")
                print("5: LaTeX table (error vs. integration)")
                print("6: LaTeX table (error vs. ground truth)")
                print("7: Error vs. Integration (average)")
                print("8: Error vs. ground truth (average)")
                print("Select plot or 'end' to go back: ", end="")
                plotNum = input()
                legend = len(simResults) > 1 and len(simResults) < 5
                if plotNum == "end":
                    terminateInner = True
                if plotNum == "0":
                    plotErrVsIntegration(simResults, legend=legend)
                elif plotNum == "1":
                    plotDirection(simResults, legend=legend)
                elif plotNum == "2":
                    plotAngularVelocity(simResults, legend=legend)
                elif plotNum == "3":
                    plotErrWithIntegration(simResults, legend=legend)
                elif plotNum == "4":
                    plotErrVsIntegrationHist(simResults)
                    pass
                elif plotNum == "5":
                    printTable(simResults)
                    pass
                elif plotNum == "6":
                    printTableGroundTruth(simResults)
                    pass
                elif plotNum == "7":
                    if len(simResults) == 1:
                        print("Unavailable, only one scenario selected")
                    else:
                        plotErrVsIntegrationAvg(simResults)
                elif plotNum == "8":
                    if len(simResults) == 1:
                        print("Unavailable, only one scenario selected")
                    else:
                        plotErrAvg(simResults)
        else:
            print("Figure '{}' unknown, exit with 'exit'".format(fig_name))
