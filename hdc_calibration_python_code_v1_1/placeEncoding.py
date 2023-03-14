import numpy as np
import math
import helper

class PositionalCueDirectionEncoder:
    # cuePos=[x,y],matDim=20 (20 rows x 20 columns),matDimReal=6 (6 unit x 6 units)
    def __init__(self, cuePos, matDim, matDimReal,FirstGlanceLRN=False):
        # Initialize Position Encoder values
        self.FirstGlanceLRN = FirstGlanceLRN
        self.posACDMatrix = np.zeros((matDim,matDim,2))
        self.cuePos = cuePos
        self.matDimReal = matDimReal
        self.scale = matDimReal/matDim
        self.oldMatPos = [None,None]
        self.firstPosAndCueVec = [[],[]] # [[x-Pos,y-Pos],[Pos->Cue Vector]]

    def checkRange(self,agentPos):
        alloVec = helper.calcVecBtw2Points(agentPos, self.cuePos)
        if (abs(alloVec[0]) < self.matDimReal / 2) and (abs(alloVec[1]) < self.matDimReal / 2):
            return True
        else:
            return False
    def getMatPos(self, agentPos):
        alloVec = helper.calcVecBtw2Points(agentPos, self.cuePos)
        mat_y = int((-alloVec[0]+self.matDimReal/2)//self.scale)
        mat_x = int((alloVec[1]+self.matDimReal/2)//self.scale)
        return [mat_x,mat_y]

    def get_set_ACDatPos(self, agentPos,DecodedACD):

            [mat_x,mat_y] = PositionalCueDirectionEncoder.getMatPos(self,agentPos)

            # if matrix indices differ from the step before: do the learning & calibration process
            if (self.oldMatPos != [mat_x,mat_y]):
                self.oldMatPos = [mat_x, mat_y]

                # Either: Learning the ACD at every Position when visiting them (FGL off)
                if self.FirstGlanceLRN == False:

                    if np.all((self.posACDMatrix[mat_x,mat_y,:] == 0)):
                        ACVec = helper.calcACVfromACD(DecodedACD)
                        self.posACDMatrix[mat_x][mat_y][:] = ACVec
                        return False

                    # if matrix already filled: get ACD
                    else:
                        return helper.calcDirFromACVec(self.posACDMatrix[mat_x][mat_y][:])

                # Or: Learn one ACV at one position and use it for calculating all ACD at all other positions (FGL on)
                else:
                    # At first glance: store ACD, agent position and cue distance
                    if len(self.firstPosAndCueVec[0])==0:
                        DecodedACVec = helper.calcACVfromACD(DecodedACD)
                        cueDist = np.linalg.norm(helper.calcVecBtw2Points(agentPos, self.cuePos))
                        self.posACDMatrix[mat_x][mat_y][:] = DecodedACVec
                        self.firstPosAndCueVec[0] = agentPos
                        self.firstPosAndCueVec[1] = np.multiply(DecodedACVec, cueDist)
                        return False
                    # Use the stored values from first glance to compute ACD when perceiving the landmark/cue again
                    else:
                        VecOldPos = helper.calcVecBtw2Points(agentPos, self.firstPosAndCueVec[0])
                        newACVec = np.add(VecOldPos, self.firstPosAndCueVec[1])
                        self.posACDMatrix[mat_x][mat_y][:] = newACVec/np.linalg.norm(newACVec)
                        return helper.calcDirFromACVec(self.posACDMatrix[mat_x][mat_y][:])
            else:
                # FGL on: Use the current position for calibration independent from previous position
                if self.FirstGlanceLRN == True:
                    VecOldPos = helper.calcVecBtw2Points(agentPos, self.firstPosAndCueVec[0])
                    newACVec = np.add(VecOldPos, self.firstPosAndCueVec[1])
                    return helper.calcDirFromACVec(newACVec/np.linalg.norm(newACVec))
                # for FGL off: if matrix indices equal the ones from the step before: do nothing
                else:
                    return False





