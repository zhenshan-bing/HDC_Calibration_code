import numpy as np
from scipy.fftpack import fft, ifft
from hdcTargetPeak import targetPeakHD
from params import n_hdc, weight_av_stim
import pickle
from neuron import phi, phi_inv, tau
from helper import loadDataFromFile

import warnings

class HDCAttractorConnectivity:
    # n: number of neurons
    # lam: lambda from Zhang 1995 paper
    # f: target activity peak function, None for default
    def __init__(self, n, lam, f=None):
        # initialize and compute F
        self.n = n
        if f == None:
            f = targetPeakHD
        self.f = f
        F = [f(i * (2*np.pi / n)) for i in range(n)]
        # compute W and store
        W = self.generateHDConnectivity(F, lam)
        self.w_hd_to_hd = W

    # get lists with weights for building the connections between the components of the calibration circuit
    def InitCalibrLoopWeights(self):
        try:
            self.w_ecd_to_conj = loadDataFromFile('data/model/weights/w_ecd_to_conj.obj')
            self.w_acd_to_conj= loadDataFromFile('data/model/weights/w_acd_to_conj.obj')
            self.w_hd_to_conj= loadDataFromFile('data/model/weights/w_hd_to_conj.obj')
            self.w_conj_to_acd= loadDataFromFile('data/model/weights/w_conj_to_acd.obj')
            self.w_conj_to_hd= loadDataFromFile('data/model/weights/w_conj_to_hd.obj')
        except Exception:
            warnings.warn("Warning:Files with calculated weights not found: Run hdcCalibrConnectivity.py before")

    # calculates n weight values for each angle difference
    def generateHDConnectivity(self, F, lam):
        # re-implemented from Zhang 1995
        # lam corresponds to lambda, the smoothness parameter

        # U directly computed from F
        U = [phi_inv(f) for f in F]

        # compute fourier transforms
        F_ = fft(F)
        U_ = fft(U)

        # compute fourier coefficients of W according to equation
        W_ = []
        for i in range(len(F_)):
            W_.append((U_[i]*F_[i]) / (lam + (abs(F_[i]))**2))

        # inverse fourier to get W
        W = ifft(W_)
        return W

    ## CONNECTIVITY FUNCTIONS FOR ALL LAYERS

    # HDC -> HDC,but also for HDC -> SHIFT & SHIFT -> HDC
    def connectionHDCtoHDC(self, i, j):
        def resolve_index(i,j,n):
            if abs(i-j)>float(n)/2.0:
                return abs(abs(i-j)-float(n))
            else:
                return abs(i-j)
        return np.real(self.w_hd_to_hd[int(resolve_index(i, j, self.n))])

    def connectionECDtoHDC(self, i, j): # hdc = i, ecd = j
        # connection weights are chosen to associate HD=0° with ECD=0°=front and HD=90° with ECD=270°=right
        if ((i+j)==0 or (i+j==n_hdc)):
            return 0.01
        else:
            return -0.0001

    def connectionECDtoCONJ(self,i,j): # conj = i, ecd = j
        return np.real(self.w_ecd_to_conj[(j+self.n-(i%self.n))%self.n])

    def connectionHDCtoCONJ(self, i, j): # conj = i, hdc = j, 0-99
        return np.real(self.w_hd_to_conj[(self.n + self.n - j - (i // self.n)) % self.n])

    def connectionCONJtoACD(self,i,j): # acd = i, conj = j
        return np.real(self.w_conj_to_acd[(self.n-(j%self.n)+(j//self.n)+(i%self.n))%self.n])

    def connectionACDtoCONJ2(self, i, j): # conj = i, acd = j, 0-99
        return np.real(self.w_acd_to_conj[(self.n+self.n-j-(i//self.n))%self.n])

    def connectionECDtoCONJ2(self, i, j):  # conj = i, ecd = j
        return np.real(self.w_ecd_to_conj[(2*self.n-j - (i % self.n)) % self.n])

    def connectionCONJ2toHDC(self, i, j):  # acd = i, conj = j
        return np.real(self.w_conj_to_hd[(self.n - (j % self.n) + (j // self.n) + (i % self.n)) % self.n])

    def connectionHDC2toHDC(self, i, j): # hdc = i, ecd = j
        return np.real(self.w_conj_to_hd[(self.n - (j % self.n) + (j // self.n) + (i % self.n)) % self.n])