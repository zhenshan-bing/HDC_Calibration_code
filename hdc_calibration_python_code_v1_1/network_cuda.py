import heapq
import multiprocessing as mp
import copy
import numpy as np
import time
import matplotlib.pyplot as plt
from neuron import tau, phi, r_m, beta_T, h_0

# pycuda
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda import gpuarray
class CudaInterface:
    def __init__(self, topology):
        self.topology = topology
        # only transfer stimuli to GPU if there were changes 
        self.currents_changed = False
        # check device attributes
        device = cuda.Device(0)
        maxThreadPerBlock = cuda.Device.get_attribute(device, cuda.device_attribute.MAX_THREADS_PER_BLOCK)
        maxThreadX = cuda.Device.get_attribute(device, cuda.device_attribute.MAX_BLOCK_DIM_X)
        self.threadLength = min([maxThreadPerBlock, maxThreadX])
        self.gridLength = cuda.Device.get_attribute(device, cuda.device_attribute.MAX_GRID_DIM_X)
        # compile cuda code
        self.registerModule()
        # save cuda functions
        self.sum_rates = self.cudaModule.get_function('sum_rates')
        self.update_rates = self.cudaModule.get_function('update_rates')
        # prepare function calls
        self.update_rates_prep = self.update_rates.prepare(["P", "P", "P", "f", "i"])
        self.sum_rates_prep = self.sum_rates.prepare(["P", "P", "P", "i", "i", "i"])
        # load network into GPU
        self.loadNetwork(topology)
        # generate execution stream
        self.stream = cuda.Stream()
    def registerModule(self):
        ########################### CUDA functions ###############################
        cudaModStr = """
            #define tau %sf
            #define r_m %sf
            #define beta_t %sf
            #define h_0 %sf
            // one thread per neuron
            __global__ void update_rates(float *rates, float *currents, float *currentsExternal, float dt, int offset) {
                // same function as in numpy implementation
                float totalInput = currents[offset + threadIdx.x] + currentsExternal[offset + threadIdx.x];
                float phi = r_m / (1 + expf(-beta_t * (totalInput - h_0)));
                rates[offset + threadIdx.x] += dt * (-rates[offset + threadIdx.x] + phi) / tau;
            }

            // one thread per neuron
            __global__ void sum_rates(float *synapses, float *rates, float *result, int fromSize, int toOffset, int fromOffset)
            {
                float sum = 0.0;
                float *syns = synapses + threadIdx.x * fromSize;
                for(int i = 0; i < fromSize; i++) {
                    //sum += rates[i + fromOffset] * synapses[threadIdx.x * fromSize + i];
                    //sum = fma(rates[i + fromOffset], synapses[threadIdx.x * fromSize + i], sum);
                    sum = fma(rates[i + fromOffset], syns[i], sum);
                }
                result[threadIdx.x + toOffset] += sum;
            }
        """ % (tau, r_m, beta_T, h_0)
        self.cudaModule = SourceModule(cudaModStr)
    def loadNetwork(self, topology):
        ##### allocate memory, save both on GPU and CPU
        # firing rates
        self.rates_cpu = np.zeros(self.topology.totalNeurons, dtype=np.float32)
        self.rates_gpu = cuda.mem_alloc(self.rates_cpu.nbytes)
        cuda.memcpy_htod(self.rates_gpu, self.rates_cpu)

        # external input currents
        self.currents_cpu = np.zeros(self.topology.totalNeurons, dtype=np.float32)
        self.currents_gpu = cuda.mem_alloc(self.currents_cpu.nbytes)
        cuda.memcpy_htod(self.currents_gpu, self.currents_cpu)
        # only transfer currents if there were changes
        self.transferCurrents = False

        # gpu array for total synaptic inputs into each neuron
        self.inputs_gpu = cuda.mem_alloc(self.rates_cpu.nbytes)

        ##### transform vector synapses to meet requirements for maximum thread count and save on gpu
        # a single thread only processes inputs for one neuron, so only split for input neurons
        vectorSynapses = []
        for entry in topology.vectorSynapses:
            ((startT, lengthT), (startF, lengthF), matrix) = entry
            offset = 0
            remaining = lengthT
            while remaining > self.threadLength:
                vectorSynapses.append(((startT + offset, self.threadLength), (startF, lengthF), matrix[offset : offset + self.threadLength]))
                remaining -= self.threadLength
                offset += self.threadLength
            if remaining != 0:
                vectorSynapses.append(((startT + offset, remaining), (startF, lengthF), matrix[offset : offset + remaining]))
        # transfer new matrices to GPU
        self.vectorSynapses = []
        for entry in vectorSynapses:
            ((startT, lengthT), (startF, lengthF), matrix) = entry
            matrix_gpu = cuda.mem_alloc(matrix.nbytes)
            cuda.memcpy_htod(matrix_gpu, matrix)
            self.vectorSynapses.append(((startT, lengthT), (startF, lengthF), matrix_gpu))
    def setCurrents(self, currents):
        self.currents_cpu = currents
        self.currents_changed = True
    def getRates(self):
        return self.rates_cpu
    # numsteps allows to do multiple steps without moving any data between CPU and GPU
    def step(self, dt, debug=False, numsteps=1):
        t_start = time.time()
        if self.currents_changed:
            cuda.memcpy_htod_async(self.currents_gpu, self.currents_cpu, stream=self.stream)
            self.currents_changed = False
        for _ in range(numsteps):
            ################# process synapse matrix ###############
            # calculate all incoming synaptic currents
            # initialize with 0
            cuda.memset_d32(self.inputs_gpu, 0, self.topology.totalNeurons)
            # run on GPU
            for entry in self.vectorSynapses:
                ((startT, lengthT), (startF, lengthF), matrix) = entry
                # self.sum_rates(matrix, self.rates_gpu, self.inputs_gpu, np.int32(lengthF), np.int32(startT), np.int32(startF), block=(lengthT, 1, 1))
                self.sum_rates.prepared_async_call((1, 1), (lengthT, 1, 1), self.stream, matrix, self.rates_gpu, self.inputs_gpu, np.int32(lengthF), np.int32(startT), np.int32(startF))
            ################# update rates #########################
            # run on gpu
            # split by thread count
            start = 0
            remaining = self.topology.totalNeurons
            while remaining > self.threadLength:
                # self.update_rates(self.rates_gpu, self.inputs_gpu, self.currents_gpu, np.float32(dt), np.int32(start), block=(self.threadLength, 1, 1))
                self.update_rates.prepared_async_call((1, 1), (self.threadLength, 1, 1), self.stream, self.rates_gpu, self.inputs_gpu, self.currents_gpu, np.float32(dt), np.int32(start))
                remaining -= self.threadLength
                start += self.threadLength
            # run remaining
            # self.update_rates(self.rates_gpu, self.inputs_gpu, self.currents_gpu, np.float32(dt), np.int32(start), block=(remaining, 1, 1))
            self.update_rates.prepared_async_call((1, 1), (remaining, 1, 1), self.stream, self.rates_gpu, self.inputs_gpu, self.currents_gpu, np.float32(dt), np.int32(start))
        # transfer result to CPU
        cuda.memcpy_dtoh_async(self.rates_cpu, self.rates_gpu, stream=self.stream)
        if debug:
            print("step time: {:.6f}ms, synapse count: {}".format(float(time.time() - t_start) * 1000.0, sum([m.shape[0] * m.shape[1] for (_, _, m) in self.topology.vectorSynapses])))
        # synchronize with stream
        self.stream.synchronize()

############################# Helper Methods #############################
# takes a list of synapse mappings and unifies them
def unionSynapses(synapsesList):
    synapses = {}
    for syn in synapsesList:
        # iterate over postsyn. neurons k
        for k in syn.keys():
            if not k in synapses:
                # no other incoming synapses registered
                synapses[k] = syn[k]
            else:
                # other incoming synapses for this neuron exist, union
                # newSynapses: mapping neuron -> strength
                newSynapses = syn[k]
                for newNeuron in newSynapses:
                    if newNeuron in synapses[k]:
                        # duplicate synapse
                        print("Duplicate synapse: {} -> {}".format(newNeuron, k))
                        # sum weights
                        synapses[k][newNeuron] += newSynapses[newNeuron]
                    else:
                        # add new synapse
                        synapses[k][newNeuron] = newSynapses[newNeuron]
    return synapses

# returns synapses connecting neurons from range1 to those of range2
#
# mapping(index1, index2): synaptic weight between 2 neurons
# index1 and index2 are in the range [0, length1] and [0, length2]
# mapping can return None for no entry 
# synapses are returned as dict mapping postsyn. neuron to another dict (presyn. neuron -> strength)
def connectLayers(start1, length1, start2, length2, mapping):
    synapses = {}
    for i in range(length1):
        synapses[i + start1] = {}
        for j in range(length2):
            synapses[i + start1][j + start2] = mapping(i, j) 
    return synapses

########################### End Helper Methods ###########################

# for building a network, when done call makeInstance to get a simulation instance
class NetworkTopology:
    def __init__(self):
        # dict mapping postsyn. neuron to another dict (presyn. neuron -> strength)
        self.synapses = {}
        # matrix M with (postsyn. layer inputs from presyn. layer) = M * (presyn. layer)
        # entries: ((postsyn. layer start, length), (presyn. layer start, length), np array matrix)
        self.vectorSynapses = []
        # mapping layer name -> (start, length)
        self.layers = {}
        self.totalNeurons = 0
    def __del__(self):
        pass
    def makeInstance(self):
        return NetworkInstance(self)
    def addLayer(self, label, length):
        self.layers[label] = (self.totalNeurons, length)
        self.totalNeurons += length
    # connect neurons from layer1 to those of layer2
    # layer1 and layer2 are layer names
    # mapping(i, j, n) returns the synaptic strength between layer1 
    def connectLayers(self, layer1, layer2, mapping):
        (start1, length1) = self.getLayer(layer1)
        (start2, length2) = self.getLayer(layer2)
        newSynapses = connectLayers(start1, length1, start2, length2, mapping)
        self.synapses = unionSynapses([self.synapses, newSynapses])
    def vectorizeConnections(self, fromLayer, toLayer):
        (startF, lengthF) = self.getLayer(fromLayer)
        (startT, lengthT) = self.getLayer(toLayer)
        matrix = np.zeros((lengthT, lengthF), dtype=np.float32)
        # put synapse dict enries into matrix
        for i in range(startT, startT + lengthT):
            if i in self.synapses.keys():
                syn = self.synapses[i]
                for j in range(startF, startF + lengthF):
                    if j in syn.keys():
                        matrix[i - startT][j - startF] = syn[j]
                        del syn[j]
                if len(syn.keys()) == 0:
                    del self.synapses[i]
        self.vectorSynapses.append(((startT, lengthT), (startF, lengthF), matrix))
    def getLayer(self, layer):
        class LayerNotFoundException(Exception):
            def __init__(self, layers, layer):
                print("##### Error: layer not found #####")
                print("Requested layer: {}".format(layer))
                layerCnt = len(layers.keys())
                if layerCnt == 0:
                    print("No layers registered")
                else:
                    if layerCnt == 1:
                        print("{} layer registered:".format(layerCnt))
                    else:
                        print("{} layers registered:".format(layerCnt))
                    for k in layers.keys():
                        print("{}: {} neurons, ids {}-{}".format(k, layers[k][1], layers[k][0], layers[k][0] + layers[k][1] - 1))
        if layer in self.layers.keys():
            return self.layers[layer]
        else:
            raise LayerNotFoundException(self.layers, layer)

class NetworkInstance:
    def __init__(self, topology):
        self.topology = topology
        self.cudaInterface = CudaInterface(self.topology)
        self.currents = np.zeros((self.topology.totalNeurons), dtype=np.float32)
    # stim: function mapping neuron index in layer to stimulus value
    def setStimulus(self, layer, stim):
        (layerStart, layerLength) = self.topology.getLayer(layer)
        for i in range(layerLength):
            self.currents[layerStart + i] = stim(i)
        self.cudaInterface.setCurrents(self.currents)
    def step(self, dt, numsteps=1):
        self.cudaInterface.step(dt, numsteps=numsteps)
    # returns rates for layer
    def getLayer(self, layer):
        (start, length) = self.topology.getLayer(layer)
        return self.cudaInterface.getRates()[start : start+length]