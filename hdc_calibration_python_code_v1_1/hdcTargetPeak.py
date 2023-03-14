import numpy as np
from params import n_hdc

# re-implemented from Zhang 1995

def targetPeak(A, B, K, x):
    return A + B*np.exp(K*np.cos(x))

K = 5.29
A = 1.72
B = 0.344

def targetPeakHD(x):
    return A + B*np.exp(K*np.cos(x))

def targetPeakCONJ(x):
    K = 5.29
    A = 0
    B = 0.0504
    return A + B*np.exp(K*np.cos(x))