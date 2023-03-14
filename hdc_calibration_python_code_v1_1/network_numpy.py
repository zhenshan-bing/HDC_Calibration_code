import heapq
import multiprocessing as mp
import copy
import numpy as np
import time
import pickle


from neuron import tau, phi, phi_vec

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
    def makeInstance(self):
        return NetworkInstance(self)
    def addLayer(self, label, length):
        self.layers[label] = (self.totalNeurons, length)
        self.totalNeurons += length
        # works with ECD until here
        #print("layers", self.layers)
    # connect neurons from layer1 to those of layer2
    # layer1 and layer2 are layer names
    # mapping(i, j, n) returns the synaptic strength between layer1 
    def connectLayers(self, layer1, layer2, mapping):
        (start1, length1) = self.getLayer(layer1)
        (start2, length2) = self.getLayer(layer2)
        newSynapses = connectLayers(start1, length1, start2, length2, mapping)
        self.synapses = unionSynapses([self.synapses, newSynapses])
        #print("synapses", self.synapses)
    def vectorizeConnections(self, fromLayer, toLayer):
        (startF, lengthF) = self.getLayer(fromLayer)
        (startT, lengthT) = self.getLayer(toLayer)
        matrix = np.zeros((lengthT, lengthF), dtype=np.float32)
        # put synapse dict entries into matrix
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
        # rates currently in use
        self.rates = np.zeros(self.topology.totalNeurons, dtype=np.float32)
        # save new rates here during simulation
        self.swaprates = np.zeros(self.topology.totalNeurons, dtype=np.float32)
        self.currents = np.zeros(self.topology.totalNeurons, dtype=np.float32)
    # stim: function mapping neuron index in layer to stimulus value
    # artificially injected stimulus onto shift layers (by Angular Vel.)
    # and on ECD Ring
    def setStimulus(self, layer, stim):
        (layerStart, layerLength) = self.topology.getLayer(layer)
        #print("layerStart, layerLength", (layerStart, layerLength))
        for i in range(layerLength):
            self.currents[layerStart + i] = stim(i)
        #print("(layerStart, layerLength)",(layerStart, layerLength),"currents",self.currents[layerStart:(layerStart+layerLength)])
    def step(self, dt, numsteps=1):
        for _ in range(numsteps):
            inCurrents = np.zeros(self.topology.totalNeurons)
            for i in range(len(self.currents)):
                inCurrents[i] = self.currents[i]
            ################# process matrix synapses ##############
            time_before = time.time()
            for entry in self.topology.vectorSynapses:
                ((startT, lengthT), (startF, lengthF), matrix) = entry
                #print("Entry---------------(startT, lengthT), (startF, lengthF)", (startT, lengthT), (startF, lengthF))
                # inCurrents[startT:startT+lengthT] += matrix.dot(self.rates[startF:startF+lengthF])

                ##############################


                """
                if (startT == 400): # access CONJ

                    filehandler = open('2_rates_conj_new.obj', 'wb')
                    pickle.dump(self.rates[startT:startT + lengthT], filehandler)
                    filehandler.close()

                    if (startF == 0):  # access HD
                        filehandler = open('1.1_matrix_HD_to_conj_new.obj', 'wb')
                        pickle.dump(matrix, filehandler)
                        filehandler.close()
                        filehandler = open('3_rates_HD_to_conj_new.obj', 'wb')
                        pickle.dump(self.rates[startF:startF+lengthF], filehandler)
                        filehandler.close()

                    if (startF == 300):  # access ECD
                        filehandler = open('1.2_matrix_ECD_to_conj_new.obj', 'wb')
                        pickle.dump(matrix, filehandler)
                        filehandler.close()
                        filehandler = open('4_rates_ECD_to_conj_new.obj', 'wb')
                        pickle.dump(self.rates[startF:startF+lengthF], filehandler)
                        filehandler.close()
                """
                inCurrents[startT:startT+lengthT] += np.matmul(matrix, self.rates[startF:startF+lengthF])
                """
                if (startT == 400): # access CONJ
                    filehandler = open('5_Currents_conj_new.obj', 'wb')
                    pickle.dump(inCurrents[startT:startT + lengthT], filehandler)
                    filehandler.close()
                """


                ##############################

            time_taken_matrix = time.time() - time_before
            # print("----------------------------!!!!!after vector processing currents", inCurrents)
            ################# process dictionary ###################
            # check if there are any synapses in the dictionary
            if self.topology.synapses:
                # print("Warning: dictionary synapses left in network, call vectorizeConnections on NetworkTopology")
                for neuron in self.topology.synapses.keys():
                    # iterate over incoming synapses
                    if neuron in self.topology.synapses.keys():
                        for fromNeuron in self.topology.synapses[neuron].keys():
                            strength = self.topology.synapses[neuron][fromNeuron]
                            inCurrents[neuron] += strength * self.rates[fromNeuron]
            ################# update rates #########################
            # explicit euler step to approximate time evolution equation from [Uli18] paper
            time_before = time.time()
            self.rates = self.rates + dt * (-self.rates + phi_vec(inCurrents)) / tau
            '''
            for neuron in range(self.topology.totalNeurons):
                self.rates[neuron] = (self.rates[neuron] + dt * (-self.rates[neuron] + phi(inCurrents[neuron])) / tau)
            '''
            time_taken_euler = time.time() - time_before
            # print(time_taken_matrix / time_taken_euler)
    # returns rates for layer
    def getLayer(self, layer):
        (start, length) = self.topology.getLayer(layer)
        return self.rates[start : start+length]
    # returns an independant network instance with only rates and currents really copied
    def copy(self):
        newInstance = NetworkInstance(self.topology)
        newInstance.currents = copy.copy(self.currents)
        newInstance.rates = copy.copy(self.rates)