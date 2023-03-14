# this file is for choosing the computation backend
# numpy is required for both
# see README.md for CUDA installation instructions

# comment out the block not used




############### CUDA with pycuda #############
from network_cuda import NetworkTopology as netTop
from network_cuda import NetworkInstance as netInst
NetworkTopology = netTop
NetworkInstance = netInst
##############################################


"""

#################### NumPy ###################
from network_numpy import NetworkTopology as netTop
from network_numpy import NetworkInstance as netInst
NetworkTopology = netTop
NetworkInstance = netInst
##############################################
"""
