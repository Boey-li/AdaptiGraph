import pybullet as p
import time
import math
from datetime import datetime

import pybullet_data


p.connect(p.GUI)
# p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
kukaId = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0], globalScaling=1.0, useFixedBase=True)
p.resetBasePositionAndOrientation(kukaId, [0, 0, 0], [0, 0, 0, 1])
kukaEndEffectorIndex = 6
numJoints = p.getNumJoints(kukaId)
if (numJoints != 7):
    exit()


p.setGravity(0, 0, -10)

useRealTimeSimulation = 0
p.setRealTimeSimulation(useRealTimeSimulation)



for i in range(1):
    print("Body %d's name is %s." % (i, p.getBodyInfo(i)[1]))

t = 0.

while 1:
    t = t + 0.1

    pos = [-0.4, 0.2 * math.cos(t), 0. + 0.2 * math.sin(t)]
    orn = p.getQuaternionFromEuler([0, -math.pi, 0])

    jointPoses = p.calculateInverseKinematics(
        kukaId, kukaEndEffectorIndex,
        pos, orn)

    #reset the joint state (ignoring all dynamics, not recommended to use during simulation)
    for i in range(numJoints):
        p.resetJointState(kukaId, i, jointPoses[i])

    print(t)
    for i in range(kukaEndEffectorIndex + 1):
        ls = p.getLinkState(kukaId, i)
        pos = ls[0]
        orn = ls[1]
        print(i, pos, orn)

