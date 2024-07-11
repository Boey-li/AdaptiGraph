import os
import numpy as np

import pybullet as p
import pybullet_data

from transformations import rotation_matrix, quaternion_from_matrix, quaternion_matrix



class KukaFleXContainer(object):

    def __init__(self):
        self.transform_bullet_to_flex = np.array([
            [1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])

        self.kuka_bullet_helper = KukaBulletContainer()

    def add_kuka(self, rest_pos, rest_orn, scaling):
        rest_pos_bullet = np.matmul(self.transform_bullet_to_flex.T[:3, :3], rest_pos)
        rest_orn_bullet = quaternion_from_matrix(
            np.matmul(self.transform_bullet_to_flex.T, quaternion_matrix(rest_orn)))

        self.kuka_bullet_helper.add_kuka(rest_pos_bullet, rest_orn_bullet, scaling)

    def get_link_state(self):
        state_pre, state_cur = self.kuka_bullet_helper.get_link_state()

        kuka_shape_states = np.zeros((8, 14))

        for i in range(8):
            kuka_shape_states[i, 0:3] = np.matmul(
                self.transform_bullet_to_flex, state_cur[i, :4])[:3]
            kuka_shape_states[i, 3:6] = np.matmul(
                self.transform_bullet_to_flex, state_pre[i, :4])[:3]
            kuka_shape_states[i, 6:10] = quaternion_from_matrix(
                np.matmul(self.transform_bullet_to_flex,
                          quaternion_matrix(state_cur[i, 4:])))
            kuka_shape_states[i, 10:14] = quaternion_from_matrix(
                np.matmul(self.transform_bullet_to_flex,
                          quaternion_matrix(state_pre[i, 4:])))

        return kuka_shape_states

    def set_ee_state(self, pos, orn):
        pos_bullet = np.matmul(self.transform_bullet_to_flex.T[:3, :3], pos)
        orn_bullet = quaternion_from_matrix(
            np.matmul(self.transform_bullet_to_flex.T, quaternion_matrix(orn)))

        self.kuka_bullet_helper.set_ee_state(pos_bullet, orn_bullet)




class KukaBulletContainer(object):

    def __init__(self, asset_root='assets'):
        asset_folder = os.path.join(asset_root, 'kuka_iiwa')

    def add_kuka(self, rest_pos=[0,0,0], rest_orn=[0,0,0,1], scaling=1.0):
        self.setup_kuka_in_pybullet(rest_pos, rest_orn, scaling)

    def setup_kuka_in_pybullet(self, pos, orn, scaling):
        # p.connect(p.DIRECT)
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.kukaId = p.loadURDF(
            "kuka_iiwa/model.urdf", pos,
            globalScaling=scaling, useFixedBase=True)
        p.resetBasePositionAndOrientation(self.kukaId, pos, orn)
        self.kukaEndEffectorIndex = 6
        self.numJoints = p.getNumJoints(self.kukaId)
        if self.numJoints != 7:
            AssertionError("numJoints != 7, %d" % self.numJoints)

        useRealTimeSimulation = 0
        p.setRealTimeSimulation(useRealTimeSimulation)

        self.numLinks = 7

        self.state_pre = np.zeros((self.numLinks + 1, 8))
        _, self.state_pre = self.get_link_state()

    def get_link_state(self):
        state_cur = []

        base_com_pos, base_com_orn = p.getBasePositionAndOrientation(self.kukaId)
        di = p.getDynamicsInfo(self.kukaId, -1)
        local_inertial_pos, local_inertial_orn = di[3], di[4]
        pos_inv, orn_inv = p.invertTransform(local_inertial_pos, local_inertial_orn)
        pos, orn = p.multiplyTransforms(base_com_pos, base_com_orn, pos_inv, orn_inv)
        state_cur.append(list(pos) + [1] + list(orn))

        for i in range(self.numLinks):
            ls = p.getLinkState(self.kukaId, i)
            pos = ls[4]
            orn = ls[5]
            state_cur.append(list(pos) + [1] + list(orn))
        return self.state_pre.copy(), np.array(state_cur).copy()

    def set_ee_state(self, pos, orn):
        _, self.state_pre = self.get_link_state()

        jointPoses = p.calculateInverseKinematics(
            self.kukaId, self.kukaEndEffectorIndex,
            pos, orn)

        for i in range(self.numJoints):
            p.resetJointState(self.kukaId, i, jointPoses[i])


