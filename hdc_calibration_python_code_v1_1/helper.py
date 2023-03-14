# Contains some commonly used helper methods
import pickle
import numpy as np

############################### Circular Topologies ##############################

# distance of neurons i and j on a circle of n neurons
# result given in "neurons"
# e.g.:
#   same neuron:        0
#   direct neighbors:   1
#   one in between:     2
def neuron_dist(i,j,n):
    if abs(i-j)>float(n)/2.0:
        return abs(abs(i-j)-float(n))
    else:
        return abs(i-j)

# absolute angle between two angles
def angleDistAbs(a, b):
    return abs(mod(((a-b) + np.pi),2*np.pi) - np.pi)

#helper to create signed modulo function
def mod(a, n):
    return a - np.floor(a/n) * n

# return angle r with b + r = a
# result signed
def angleDist(b, a):
    # before
    b = b % (2*np.pi)
    e_b = b if b < np.pi else b - 2*np.pi
    # after
    a = a % (2*np.pi)
    e_a = a if a < np.pi else a - 2*np.pi
    # fix transitions pi <=> -pi
    # in top left quadrant
    e_b_topleft = e_b < np.pi and e_b > np.pi / 2
    e_a_topleft = e_a < np.pi and e_a > np.pi / 2
    # in bottom left quadrant
    e_b_bottomleft = e_b < -np.pi / 2 and e_b > -np.pi
    e_a_bottomleft = e_a < -np.pi / 2 and e_a > -np.pi
    if e_a_topleft and e_b_bottomleft:
        # transition in negative direction
        return -(abs(e_a - np.pi) + abs(e_b + np.pi))
    elif e_a_bottomleft and e_b_topleft:
        # transition in positive direction
        return abs(e_a + np.pi) + abs(e_b - np.pi)
    else:
        # no transition, just the difference
        return e_a - e_b

# average of all the neuron's preferred directions weighted their corresponding activities
# neuron 0 corresponds to 0 deg, all neurons are equally spaced on the circle in ascending order
def decodeAttractorNumpy(activity):
    n = len(activity)
    # make a vector for every cell, pointing in its preferred direction
    angles = np.array([(2. * np.pi) * (float(i) / n) for i in range(n)])
    x_components = np.cos(angles)
    y_components = np.sin(angles)
    # scale those by cell's activity, normalize by sum of activities
    actsum = sum(activity)
    x_components *= (activity / actsum)
    y_components *= (activity / actsum)
    # sum over vectors
    vec = np.array([np.sum(x_components), np.sum(y_components)])
    # transform back to angles
    result = -np.arctan2(vec[1], vec[0])
    if result < 0:
        result = 2 * np.pi + result
    return 2*np.pi - result

# average of all the neuron's preferred directions weighted their corresponding activities
# neuron 0 corresponds to 0 deg, all neurons are equally spaced on the circle in ascending order
def decodeRingActivity(activity):
    n = len(activity)
    # make a vector for every cell, pointing in its tuned direction
    angles = [(2. * np.pi) * (float(i) / n) for i in range(n)]
    vectors = [np.array([ np.cos(angles[i]), np.sin(angles[i]) ]) for i in range(n)]
    # scale those by cell's activity, normalize by sum of activities
    actsum = sum(activity)
    for i in range(n):
        vectors[i] *= activity[i] / actsum
    # sum over vectors
    vec = np.array([0.0, 0.0])
    for i in range(n):
        vec += vectors[i]
    # transform back to angle
    result = -np.arctan2(vec[1], vec[0])
    if result < 0:
        result = 2 * np.pi + result
    return 2*np.pi - result

# transform angles from [0, 2pi) to [-pi, pi), re-ordering X
# Y is transformed to correspond to the new positions
def centerAnglesWithY(X, Y):
    newX = [x if x < np.pi else (x - 2*np.pi) for x in X]
    XY = list(zip(newX, Y))
    XY.sort()
    newX, newY = zip(*XY)
    return list(newX), list(newY)

def radToDeg(X):
    return [x * (360 / (2*np.pi)) for x in X]

def calcVecBtw2Points(FromPoint, ToPoint):
    alloCueVec = np.subtract(np.array(ToPoint), np.array(FromPoint[:2]))
    return alloCueVec

def calcDirFromACVec(alloCueVec):
    alloCueDir = np.angle([complex(alloCueVec[0], alloCueVec[1])])
    if (alloCueDir < 0.0):
        alloCueDir += (2 * np.pi)
    return alloCueDir

def calcACDir(agentPos, cuePos):
    alloCueVec = calcVecBtw2Points(np.array(agentPos[:2]), np.array(cuePos))
    return calcDirFromACVec(alloCueVec)

def calcACVfromACD(ACD):
    return np.array([np.cos(ACD),np.sin(ACD)])

def calcECDir(agentDir,ACDir):
    diff = ACDir - agentDir
    if (diff>=0.0):
        return diff
    else:
        return diff+2*np.pi

def cueInSight(viewAngle,ECDir):
    if ((viewAngle/2)<ECDir<((2*np.pi)-(viewAngle/2))):
        return False
    else:
        return True

def loadDataFromFile(fileName):
    filehandler = open(fileName, 'rb')
    loadedFile = (pickle.load(filehandler))
    filehandler.close()
    return loadedFile

def writeDataToFile(fileToDump,fileName):
    filehandler = open(fileName, 'wb')
    pickle.dump(fileToDump, filehandler)
    filehandler.close()

