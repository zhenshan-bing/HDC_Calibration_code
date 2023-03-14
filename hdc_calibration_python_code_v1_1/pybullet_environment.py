import pybullet as p
import gym
import signal
import sys
import pybullet_data
import math
import numpy as np
from scipy.spatial.transform import Rotation as R
import time


def signal_handler(signal, frame):
    print("\nprogram exiting gracefully")
    sys.exit(0)

def angleDist(a, b):
    reala = a % (2 * np.pi)
    realb = b % (2 * np.pi)
    return min(abs(reala - realb), abs((reala + 2 * np.pi) - realb))
    #return min(abs(reala - realb), abs(reala - (2*np.pi - realb)))

class PybulletEnvironment(gym.Env):
    def __init__(self, rate, visualize, model, cuePos):
        self.model = model
        self.visualize = visualize
        self.rate_ = rate
        # connect to pybullet
        if self.visualize:
            p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        else:
            p.connect(p.DIRECT)
        p.setTimeStep(1.0/self.rate_)  # default is 240 Hz
        # reset camera angle
        if self.model=="maze":
            p.resetDebugVisualizerCamera(cameraDistance=4.5, cameraYaw=0, cameraPitch=-70, cameraTargetPosition=[0.55, -7.35, 5.0])
        elif self.model == "plus":
            p.resetDebugVisualizerCamera(cameraDistance=4.5, cameraYaw=0, cameraPitch=-70, cameraTargetPosition=[0.55, -0.35, 0.2])
        elif self.model == "box":
            p.resetDebugVisualizerCamera(cameraDistance=2.5, cameraYaw=0, cameraPitch=-70,
                                         cameraTargetPosition=[0.55, -0.35, 0.2])
        self.carId = []
        self.planeId = []
        self.cueId = []
        self.cuePos = cuePos
        self.agentState =  {'Position': None,'Orientation': None}
        self.action_space = []
        self.observation_space = []
        self.euler_angle = 0
        self.euler_angle_before = 0
        pass

    def step(self, action):
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING, 1, p.COV_ENABLE_GUI, 0)
        self.euler_angle_before = self.euler_angle
        self.braitenberg()

        #Get Position
        posAndOr = p.getBasePositionAndOrientation(self.carId) #posAndOr ((-0.05, -2.0, 0.02), (0.0, 0.0, 0.0, 1.0))
        self.agentState['Position'] = posAndOr[0][:2]
        self.agentState['Orientation'] = p.getEulerFromQuaternion(posAndOr[1])[2]
        if (self.agentState['Orientation'] < 0): self.agentState['Orientation'] += 2*np.pi
        # step simulation
        p.stepSimulation()

        # return change in orientation
        # before
        e_b = self.euler_angle_before[2]
        # after
        e_a = self.euler_angle[2]
        # fix transitions pi <=> -pi
        # in top left quadrant
        e_b_topleft = e_b < np.pi and e_b > np.pi / 2
        e_a_topleft = e_a < np.pi and e_a > np.pi / 2
        # in bottom left quadrant
        e_b_bottomleft = e_b < -np.pi / 2 and e_b > -np.pi
        e_a_bottomleft = e_a < -np.pi / 2 and e_a > -np.pi
        if e_a_topleft and e_b_bottomleft:
            # transition in negative direction
            return -(abs(e_a - np.pi) + abs(e_b + np.pi)),self.agentState
        elif e_a_bottomleft and e_b_topleft:
            # transition in positive direction
            return (abs(e_a + np.pi) + abs(e_b - np.pi)),self.agentState
        else:
            # no transition, just the difference
            return (e_a - e_b),self.agentState

    def reset(self):
        # reset simulation
        p.resetSimulation()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)  # we will enable rendering after we loaded everything
        p.setGravity(0, 0, -10 * 1)

        # reload model
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        if self.model == "maze":
            self.planeId = p.loadURDF("maze_2_2_lane/plane.urdf")
            self.cueId = p.loadURDF("p3dx/cue/visual_cue.urdf", basePosition=self.cuePos)
            self.carId = p.loadURDF("p3dx/urdf/pioneer3dx.urdf", basePosition=[8.0, -10, 0.02], baseOrientation=[0.0, 0.0, 0.7071067811865475, 0.7071067811865476])
            cubeStartOrientation = p.getQuaternionFromEuler([0, 0, np.pi/2])
        elif self.model == "plus":
            self.planeId = p.loadURDF("p3dx/plane/plane.urdf")
            self.cueId = p.loadURDF("p3dx/cue/visual_cue.urdf", basePosition=self.cuePos)
            self.carId = p.loadURDF("p3dx/urdf/pioneer3dx.urdf", basePosition=[-1.5, -2, 0.02])#[0, -2, 0.02])
        elif self.model == "box":
            self.planeId = p.loadURDF("p3dx/plane/plane_box.urdf")
            self.cueId = p.loadURDF("p3dx/cue/visual_cue.urdf", basePosition=self.cuePos)
            self.carId = p.loadURDF("p3dx/urdf/pioneer3dx.urdf", basePosition=[0, -0.5, 0.02])
        self.euler_angle = p.getEulerFromQuaternion(p.getLinkState(self.carId, 0)[1])
        #print("Euler_angle_before: ", self.euler_angle_before)

        # render back
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)  # rendering's back on again

        observation = []
        return observation

    def render(self, mode='human'):
        pass

    def close(self):
        p.disconnect()

    def braitenberg(self):
        detect = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        if self.model=="maze" or self.model=="box":
            braitenbergL = np.array(
                [-0.8, -0.6, -0.4, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.6, -1.4, -1.2, -1.0])
            braitenbergR = np.array(
                [-1.0, -1.2, -1.4, -1.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.2, -0.4, -0.6, -0.8])
            noDetectionDist = 1.75
            velocity_0 = 5.5
            maxDetectionDist = 0.25
        else:
            braitenbergL = np.array(
                [-0.8, -0.6, -0.4, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.6, -1.4, -1.2, -1.0])
            braitenbergR = np.array(
                [-1.0, -1.2, -1.4, -1.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.2, -0.4, -0.6, -0.8])
            noDetectionDist = 1.0
            velocity_0 = 2.0
            maxDetectionDist = 0.2

        rayDist = self.ray_detection()
        # print("Ray Dist: ", rayDist)

        for i in range(len(rayDist)):
            if 0 < rayDist[i] < noDetectionDist:
                # something is detected
                if rayDist[i] < maxDetectionDist:
                    rayDist[i] = maxDetectionDist
                # dangerous level, the higher, the closer
                detect[i] = 1.0 - 1.0 * ((rayDist[i] - maxDetectionDist) * 1.0 / (noDetectionDist - maxDetectionDist))
            else:
                # nothing is detected
                detect[i] = 0

        vLeft = velocity_0
        vRight = velocity_0

        # print(detect)
        for i in range(len(rayDist)):
            vLeft = vLeft + braitenbergL[i] * detect[i] * 1
            vRight = vRight + braitenbergR[i] * detect[i] * 1

        '''
        minVelocity = 0.5
        if abs(vLeft) < minVelocity and abs(vRight) < minVelocity:
            vLeft = minVelocity
            vRight = minVelocity
        print("V Left:", vLeft, "V Right", vRight)
        '''
        p.setJointMotorControlArray(bodyUniqueId=self.carId,
                                    jointIndices=[4, 6],
                                    controlMode=p.VELOCITY_CONTROL,
                                    targetVelocities=[vLeft, vRight],
                                    forces=[10, 10])

    def ray_detection(self):
        # the index of the ray is from the front, counter-clock-wise direction #
        # detect range rayLen = 1 #
        p.removeAllUserDebugItems()

        rayReturn = []
        rayFrom = []
        rayTo = []
        rayIds = []
        numRays = 16
        if self.model=="maze":
            rayLen = 1.75
        else:
            rayLen = 1
        rayHitColor = [1, 0, 0]
        rayMissColor = [0, 1, 0]
        replaceLines = True

        for i in range(numRays):
            # rayFromPoint = p.getBasePositionAndOrientation(self.carId)[0]
            rayFromPoint = p.getLinkState(self.carId, 0)[0]
            rayReference = p.getLinkState(self.carId, 0)[1]
            euler_angle = p.getEulerFromQuaternion(rayReference)  # in degree
            # print("Euler Angle: ", rayFromPoint)
            rayFromPoint = list(rayFromPoint)
            rayFromPoint[2] = rayFromPoint[2] + 0.02
            rayFrom.append(rayFromPoint)
            rayTo.append([
                rayLen * math.cos(
                    2.0 * math.pi * float(i) / numRays + 360.0 / numRays / 2 * math.pi / 180 + euler_angle[2]) +
                rayFromPoint[0],
                rayLen * math.sin(
                    2.0 * math.pi * float(i) / numRays + 360.0 / numRays / 2 * math.pi / 180 + euler_angle[2]) +
                rayFromPoint[1],
                rayFromPoint[2]
            ])

            # if (replaceLines):
            #     if i == 0:
            #         # rayIds.append(p.addUserDebugLine(rayFrom[i], rayTo[i], [0, 0, 1]))
            #         pass
            #     else:
            #         # rayIds.append(p.addUserDebugLine(rayFrom[i], rayTo[i], rayMissColor))
            #         pass
            # else:
            #     rayIds.append(-1)

        results = p.rayTestBatch(rayFrom, rayTo, numThreads=0)
        for i in range(numRays):
            hitObjectUid = results[i][0]

            if (hitObjectUid < 0):
                hitPosition = [0, 0, 0]
                # p.addUserDebugLine(rayFrom[i], rayTo[i], rayMissColor, replaceItemUniqueId=rayIds[i])
                p.addUserDebugLine(rayFrom[i], rayTo[i], rayMissColor)
                rayReturn.append(-1)
            else:
                hitPosition = results[i][3]
                # p.addUserDebugLine(rayFrom[i], hitPosition, rayHitColor, replaceItemUniqueId=rayIds[i])
                p.addUserDebugLine(rayFrom[i], rayTo[i], rayHitColor)
                rayReturn.append(
                    math.sqrt((hitPosition[0] - rayFrom[i][0]) ** 2 + (hitPosition[1] - rayFrom[i][1]) ** 2))

        self.euler_angle = euler_angle
        # print("euler_angle: ", euler_angle[2] * 180 / np.pi)
        return rayReturn

    def euler_calculation(self):
        position, orientation = p.getBasePositionAndOrientation(self.carId)
        r = R.from_quat(list(orientation))
        euler_angle = r.as_euler('zyx', degrees=True)
        print("orientation: ", euler_angle)
        pass
