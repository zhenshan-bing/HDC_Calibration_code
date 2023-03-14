import numpy as np
###################### built from [Uli18] paper equations ############################

# parameters of sigmoit function phi
# r_m: max rate
r_m = 76.2
# beta_T: slope at inflection point
beta_T = 0.82
# h_0: threshold (current at inflection point)
h_0 = 2.46

# time constant in seconds
tau = 0.020

def phi(x):
    return r_m/(1 + np.exp(-beta_T * (x - h_0)))

def phi_vec(x):
    return r_m * np.reciprocal(1.0 + np.exp(-beta_T * (x - h_0)))

def phi_inv(y):
    return h_0 + np.log(-y / (y - r_m)) / beta_T

# time evolution: tau * (du / dt) = -rate_i + phi(incoming currents to neuron i)
# simulated in NetworkWorkload with explicit euler

########################### [Uli18] paper stop ##################################