from typing import Optional
import sys
sys.path.append('.')
import os

import cv2
import json
import time
import pickle
import numpy as np
import torch
import math
import copy

from multiprocessing.managers import SharedMemoryManager
from real_world.camera.multi_realsense import MultiRealsense, SingleRealsense
from real_world.xarm6 import XARM6
from real_world.utils import depth2fgpcd, rpy_to_rotation_matrix, similarity_transform
from plan_utils import decode_action_single


class RealEnv:
    def __init__(self, 
            task_config=None,
            WH=[640, 480],
            capture_fps=15,
            obs_fps=15,
            n_obs_steps=1,
            enable_color=True,
            enable_depth=True,
            process_depth=False,
            use_robot=True,
            verbose=False,
            gripper_enable=False,
            speed=50,
            wrist=None,
        ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.WH = WH
        self.capture_fps = capture_fps
        self.obs_fps = obs_fps
        self.n_obs_steps = n_obs_steps
        if wrist is None:
            print('No wrist camera. Using default camera id.')
            self.WRIST = '311322300308'
        else:
            self.WRIST = wrist

        base_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        self.vis_dir = os.path.join(base_path, 'dump/vis_real_world')

        self.serial_numbers = SingleRealsense.get_connected_devices_serial()
        if self.WRIST is not None and self.WRIST in self.serial_numbers:
            print('Found wrist camera.')
            self.serial_numbers.remove(self.WRIST)
            self.serial_numbers = self.serial_numbers + [self.WRIST]  # put the wrist camera at the end
            self.n_fixed_cameras = len(self.serial_numbers) - 1
        else:
            self.n_fixed_cameras = len(self.serial_numbers)
        print(f'Found {self.n_fixed_cameras} fixed cameras.')

        self.shm_manager = SharedMemoryManager()
        self.shm_manager.start()

        self.realsense =  MultiRealsense(
                serial_numbers=self.serial_numbers,
                shm_manager=self.shm_manager,
                resolution=(self.WH[0], self.WH[1]),
                capture_fps=self.capture_fps,
                enable_color=enable_color,
                enable_depth=enable_depth,
                process_depth=process_depth,
                verbose=verbose)
        self.realsense.set_exposure(exposure=100, gain=60)
        self.realsense.set_white_balance()
        self.last_realsense_data = None
        self.enable_color = enable_color
        self.enable_depth = enable_depth
        self.use_robot = use_robot

        if use_robot:
            self.robot = XARM6(gripper_enable=gripper_enable, speed=speed)
        self.gripper_enable = gripper_enable

        calibration_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        calibration_parameters =  cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(calibration_dictionary, calibration_parameters)
        self.calibration_board = cv2.aruco.GridBoard(
            (5, 7),
            markerLength=0.03435,
            markerSeparation=0.00382,
            dictionary=calibration_dictionary,
        )

        self.R_cam2world = None
        self.t_cam2world = None
        self.R_base2world = None
        self.t_base2world = None
        
        self.calibrate_result_dir = os.path.join(base_path, 'dump/calibration_result')
        os.makedirs(self.calibrate_result_dir, exist_ok=True)

        self.task_config = task_config
        if task_config is not None:
            pusher_points = []
            for item in task_config['pusher_points']:
                pusher_points.append(np.array(item))
            self.pusher_points = np.array(pusher_points)
            self.bbox = np.array([
                [float(task_config['bbox'][0]), float(task_config['bbox'][1])], 
                [float(task_config['bbox'][2]), float(task_config['bbox'][3])], 
                [float(task_config['bbox'][4]), float(task_config['bbox'][5])]])
            self.clipping_height = task_config['clipping_height']  # min z to prevent colliding with the table
        else:
            print('No task config.')
            self.pusher_points = None
            self.bbox = np.array([[-0.45, 0.0], [-0.25, 0.45], [-0.2, 0.05]])  # default
            self.clipping_height = None

        self.pusher_height = 0.008  # pusher's lowest point; hard coded

    # ======== start-stop API =============
    @property
    def is_ready(self):
        return self.realsense.is_ready and (self.robot.is_alive if self.use_robot else True)
    
    def start(self, wait=True, exposure_time=5):
        self.realsense.start(wait=False, put_start_time=time.time() + exposure_time)
        if wait:
            self.start_wait()

    def stop(self, wait=True):
        self.realsense.stop(wait=False)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.realsense.start_wait()
    
    def stop_wait(self):
        self.realsense.stop_wait()
    
    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= async env API ===========
    def get_obs(self, get_color=True, get_depth=False) -> dict:
        assert self.is_ready

        # get data
        k = math.ceil(self.n_obs_steps * (self.capture_fps / self.obs_fps))
        self.last_realsense_data = self.realsense.get(
            k=k,
            out=self.last_realsense_data
        )

        robot_obs = dict()
        if self.use_robot:
            robot_obs['joint_angles'] = self.robot.get_current_joint()
            robot_obs['pose'] = self.robot.get_current_pose()
            if self.gripper_enable:
                robot_obs['gripper_position'] = self.robot.get_gripper_state()

        # align camera obs timestamps
        dt = 1 / self.obs_fps
        timestamp_list = [x['timestamp'][-1] for x in self.last_realsense_data.values()]
        last_timestamp = np.max(timestamp_list)
        obs_align_timestamps = last_timestamp - (np.arange(self.n_obs_steps)[::-1] * dt)
        # the last timestamp is the latest one

        camera_obs = dict()
        for camera_idx, value in self.last_realsense_data.items():
            this_timestamps = value['timestamp']
            this_idxs = list()
            for t in obs_align_timestamps:
                is_before_idxs = np.nonzero(this_timestamps < t)[0]
                this_idx = 0
                if len(is_before_idxs) > 0:
                    this_idx = is_before_idxs[-1]
                this_idxs.append(this_idx)
            # remap key
            if get_color:
                assert self.enable_color
                camera_obs[f'color_{camera_idx}'] = value['color'][this_idxs]
            if get_depth and isinstance(camera_idx, int):
                assert self.enable_depth
                camera_obs[f'depth_{camera_idx}'] = value['depth'][this_idxs] / 1000.0

        # return obs
        obs_data = dict(camera_obs)
        obs_data.update(robot_obs)
        obs_data['timestamp'] = obs_align_timestamps
        return obs_data

    def get_intrinsics(self):
        return self.realsense.get_intrinsics()

    def get_extrinsics(self):
        return (
            [self.R_cam2world[i].copy() for i in self.serial_numbers[:4]],
            [self.t_cam2world[i].copy() for i in self.serial_numbers[:4]],
        )

    def get_bbox(self):
        return self.bbox.copy()

    def step(self, action, decoded=False):
        assert self.use_robot
        assert self.is_ready

        if decoded:
            x_start, z_start, x_end, z_end = action
        else:
            x_start, z_start, x_end, z_end = decode_action_single(action, push_length=self.task_config['push_length'])

        # coordinate transform
        x_start /= self.task_config['sim_real_ratio']
        z_start /= self.task_config['sim_real_ratio']
        x_end /= self.task_config['sim_real_ratio']
        z_end /= self.task_config['sim_real_ratio']

        if self.task_config['rotate_pusher']:
            yaw = 180 - np.arctan2(z_end - z_start, x_end - x_start) / np.pi * 180  # rotatable pusher
        else:
            yaw = None  # stick pusher

        self.reset_robot()
        time.sleep(0.5)
        self.move_to_table_position([x_start, self.pusher_height + 0.15, z_start], yaw, wait=True)
        self.move_to_table_position([x_start, self.pusher_height, z_start], yaw, wait=True)
        time.sleep(0.5)
        self.move_to_table_position([x_end, self.pusher_height, z_end], yaw, wait=True)
        self.move_to_table_position([x_end, self.pusher_height + 0.15, z_end], yaw, wait=True)
        time.sleep(0.5)
        self.reset_robot()
    
    def step_gripper(self, action, decoded=False):
        assert self.use_robot
        assert self.gripper_enable
        assert self.is_ready
        if decoded:
            x_start, z_start, x_end, z_end = action
        else:
            x_start, z_start, x_end, z_end = decode_action_single(action, push_length=self.task_config['push_length'])
        
        # coordinate transform
        x_start /= self.task_config['sim_real_ratio']
        z_start /= self.task_config['sim_real_ratio']
        x_end /= self.task_config['sim_real_ratio']
        z_end /= self.task_config['sim_real_ratio']

        if self.task_config['rotate_pusher']:
            yaw = 180 - np.arctan2(z_end - z_start, x_end - x_start) / np.pi * 180  # rotatable pusher
        else:
            yaw = None  # stick pusher

        x_start = x_start - 0.005 * (x_end - x_start) / np.sqrt((x_end - x_start) ** 2 + (z_end - z_start) ** 2)
        z_start = z_start - 0.005 * (z_end - z_start) / np.sqrt((x_end - x_start) ** 2 + (z_end - z_start) ** 2)

        self.reset_robot()
        self.move_to_table_position([x_start, self.pusher_height + 0.15, z_start], yaw, wait=True)
        self.move_to_table_position([x_start, self.pusher_height, z_start], yaw, wait=True)
        time.sleep(5)
        self.robot.close_gripper()
        time.sleep(0.5)
        self.move_to_table_position([x_start, self.pusher_height + 0.02, z_start], yaw, wait=True)
        self.move_to_table_position([x_end, self.pusher_height + 0.02, z_end], yaw, wait=True)
        self.robot.open_gripper()
        time.sleep(0.5)
        self.move_to_table_position([x_end, self.pusher_height + 0.15, z_end], yaw, wait=True)
        self.reset_robot()

    def move_to_table_position(self, position, yaw=None, wait=True):
        assert self.use_robot
        assert self.is_ready
        if yaw:
            # perpendicular to the pushing direction
            rpy = np.array([180., 0., yaw])
        else:
            rpy = np.array([180., 0., 0.])
        R_gripper2base = rpy_to_rotation_matrix(rpy[0], rpy[1], rpy[2])

        # (x, -z, y) to (x, y, z)
        position = np.array(position)
        position = np.array([position[0], position[2], -position[1]])

        R_base2world = self.R_base2world
        t_base2world = self.t_base2world

        R_world2base = R_base2world.T
        t_world2base = -R_base2world.T @ t_base2world

        finger_in_world = position
        finger_in_base = R_world2base @ finger_in_world + t_world2base

        if finger_in_base[2] < self.clipping_height:
            print("clipping the stick pose to not collide with the table")
            finger_in_base[2] = self.clipping_height

        gripper_in_base = finger_in_base - R_gripper2base @ self.pusher_points[0]

        pose = np.concatenate([gripper_in_base * 1000, rpy], axis=0)
        self.robot.move_to_pose(pose=pose, wait=wait, ignore_error=True)

    def get_robot_pose(self, raw=False):
        raw_pose = self.robot.get_current_pose()
        if raw:
            return raw_pose
        else:
            R_gripper2base = rpy_to_rotation_matrix(
                raw_pose[3], raw_pose[4], raw_pose[5]
            )
            t_gripper2base = np.array(raw_pose[:3]) / 1000
        return R_gripper2base, t_gripper2base

    def set_robot_pose(self, pose, wait=True):
        self.robot.move_to_pose(pose=pose, wait=wait, ignore_error=True)
    
    def reset_robot(self, wait=True):
        self.robot.reset(wait=wait)
    
    def hand_eye_calibrate(self, visualize=True, save=True, return_results=True):
        self.reset_robot()
        time.sleep(1)

        poses = [
            [522.6, -1.6, 279.5, 179.2, 0, 0.3],
            [494.3, 133, 279.5, 179.2, 0, -24.3],
            [498.8, -127.3, 314.9, 179.3, 0, 31.1],
            [589.5, 16.6, 292.9, -175, 17, 1.2],
            [515.8, 178.5, 469.2, -164.3, 17.5, -90.8],
            [507.9, -255.5, 248.5, -174.6, -16.5, 50.3],
            [507.9, 258.2, 248.5, -173.5, -8, -46.8],
            [569, -155.6, 245.8, 179.5, 3.7, 49.7],
            [570.8, -1.2, 435, -178.5, 52.3, -153.9],
            [474.3, 12.5, 165.3, 179.3, -15, 0.3],
        ]
        R_gripper2base = []
        t_gripper2base = []
        R_board2cam = []
        t_board2cam = []

        if visualize:
            os.makedirs(f'{self.vis_dir}', exist_ok=True)
        
        for pose in poses:
            # Move to the pose and wait for 5s to make it stable
            self.set_robot_pose(pose)
            time.sleep(5)

            # Calculate the markers
            obs = self.get_obs()

            pose_real = obs['pose']
            calibration_img = obs[f'color_{self.n_fixed_cameras}'][-1]

            intr = self.get_intrinsics()[-1]
            dist_coef = np.zeros(5)

            if visualize:
                cv2.imwrite(f'{self.vis_dir}/calibration_handeye_img_{pose}.jpg', calibration_img)

            calibration_img = cv2.cvtColor(calibration_img, cv2.COLOR_BGR2GRAY)

            # calibrate
            corners, ids, rejected_img_points = self.aruco_detector.detectMarkers(calibration_img)
            detected_corners, detected_ids, rejected_corners, recovered_ids = self.aruco_detector.refineDetectedMarkers(
                detectedCorners=corners, 
                detectedIds=ids,
                rejectedCorners=rejected_img_points,
                image=calibration_img,
                board=self.calibration_board,
                cameraMatrix=intr,
                distCoeffs=dist_coef,
            )

            if visualize:
                calibration_img_vis = cv2.aruco.drawDetectedMarkers(calibration_img.copy(), detected_corners, detected_ids)
                cv2.imwrite(f'{self.vis_dir}/calibration_marker_handeye_{pose}.jpg', calibration_img_vis)

            retval, rvec, tvec = cv2.aruco.estimatePoseBoard(
                corners=detected_corners,
                ids=detected_ids,
                board=self.calibration_board,
                cameraMatrix=intr,
                distCoeffs=dist_coef,
                rvec=None,
                tvec=None,
            )

            if visualize:
                calibration_img_vis = calibration_img.copy()[:, :, np.newaxis].repeat(3, axis=2)
                cv2.drawFrameAxes(calibration_img_vis, intr, dist_coef ,rvec, tvec, 0.1)
                cv2.imwrite(f"{self.vis_dir}/calibration_result_handeye_{pose}.jpg", calibration_img_vis)

            if not retval:
                raise ValueError("pose estimation failed")

            # Save the transformation of board2cam
            R_board2cam.append(cv2.Rodrigues(rvec)[0])
            t_board2cam.append(tvec[:, 0])

            # Save the transformation of the gripper2base
            print("Current pose: ", pose_real)

            R_gripper2base.append(
                rpy_to_rotation_matrix(
                    pose_real[3], pose_real[4], pose_real[5]
                )
            )
            t_gripper2base.append(np.array(pose_real[:3]) / 1000)
        
        self.reset_robot()

        R_base2gripper = []
        t_base2gripper = []
        for i in range(len(R_gripper2base)):
            R_base2gripper.append(R_gripper2base[i].T)
            t_base2gripper.append(-R_gripper2base[i].T @ t_gripper2base[i])

        # Do the robot-world hand-eye calibration
        R_base2world, t_base2world, R_gripper2cam, t_gripper2cam = cv2.calibrateRobotWorldHandEye(
            R_world2cam=R_board2cam,
            t_world2cam=t_board2cam,
            R_base2gripper=R_base2gripper,
            t_base2gripper=t_base2gripper,
            R_base2world=None,
            t_base2world=None,
            R_gripper2cam=None,
            t_gripper2cam=None,
            method=cv2.CALIB_HAND_EYE_TSAI,
        )

        t_gripper2cam = t_gripper2cam[:, 0]  # (3, 1) -> (3,)
        t_base2world = t_base2world[:, 0]  # (3, 1) -> (3,)

        results = {}
        results["R_gripper2cam"] = R_gripper2cam
        results["t_gripper2cam"] = t_gripper2cam
        results["R_base2world"] = R_base2world
        results["t_base2world"] = t_base2world

        print("R_gripper2cam", R_gripper2cam)
        print("t_gripper2cam", t_gripper2cam)
        if save:
            with open(f"{self.calibrate_result_dir}/calibration_handeye_result.pkl", "wb") as f:
                pickle.dump(results, f)
        if return_results:
            return results

    def fixed_camera_calibrate(self, visualize=True, save=True, return_results=True):
        if visualize:
            os.makedirs(f'{self.vis_dir}', exist_ok=True)
        
        rvecs = {}
        tvecs = {}
        rvecs_list = []
        tvecs_list = []

        # Calculate the markers
        obs = self.get_obs()
        intrs = self.get_intrinsics()
        dist_coef = np.zeros(5)

        for i in range(self.n_fixed_cameras):  # ignore the wrist camera
            device = self.serial_numbers[i]
            intr = intrs[i]
            calibration_img = obs[f'color_{i}'][-1].copy()
            if visualize:
                cv2.imwrite(f'{self.vis_dir}/calibration_img_{device}.jpg', calibration_img)
            
            calibration_img = cv2.cvtColor(calibration_img, cv2.COLOR_BGR2GRAY)

            corners, ids, rejected_img_points = self.aruco_detector.detectMarkers(calibration_img)
            detected_corners, detected_ids, rejected_corners, recovered_ids = self.aruco_detector.refineDetectedMarkers(
                detectedCorners=corners, 
                detectedIds=ids,
                rejectedCorners=rejected_img_points,
                image=calibration_img,
                board=self.calibration_board,
                cameraMatrix=intr,
                distCoeffs=dist_coef,
            )

            if visualize:
                calibration_img_vis = cv2.aruco.drawDetectedMarkers(calibration_img.copy(), detected_corners, detected_ids)
                cv2.imwrite(f'{self.vis_dir}/calibration_detected_marker_{device}.jpg', calibration_img_vis)


            retval, rvec, tvec = cv2.aruco.estimatePoseBoard(
                corners=detected_corners,
                ids=detected_ids,
                board=self.calibration_board,
                cameraMatrix=intr,
                distCoeffs=dist_coef,
                rvec=None,
                tvec=None,
            )

            if not retval:
                print("pose estimation failed")
                import ipdb; ipdb.set_trace()

            if visualize:
                calibration_img_vis = calibration_img.copy()[:, :, np.newaxis].repeat(3, axis=2)
                cv2.drawFrameAxes(calibration_img_vis, intr, dist_coef, rvec, tvec, 0.1)
                cv2.imwrite(f"{self.vis_dir}/calibration_result_{device}.jpg", calibration_img_vis)

            rvecs[device] = rvec
            tvecs[device] = tvec
            rvecs_list.append(rvec)
            tvecs_list.append(tvec)
        
        if save:
            # save rvecs, tvecs
            with open(f'{self.calibrate_result_dir}/rvecs.pkl', 'wb') as f:
                pickle.dump(rvecs, f)
            with open(f'{self.calibrate_result_dir}/tvecs.pkl', 'wb') as f:
                pickle.dump(tvecs, f)
            
            # save rvecs, tvecs, intrinsics as numpy array
            rvecs_list = np.array(rvecs_list)
            tvecs_list = np.array(tvecs_list)
            intrs = np.array(intrs)
            with open(f'{self.calibrate_result_dir}/rvecs.npy', 'wb') as f:
                np.save(f, rvecs_list)
            with open(f'{self.calibrate_result_dir}/tvecs.npy', 'wb') as f:
                np.save(f, tvecs_list)
            with open(f'{self.calibrate_result_dir}/intrinsics.npy', 'wb') as f:
                np.save(f, intrs)

        if return_results:
            return rvecs, tvecs

    def calibrate(self, re_calibrate=False):
        if re_calibrate:
            if self.use_robot:
                calibration_handeye_result = self.hand_eye_calibrate()
                R_base2board = calibration_handeye_result['R_base2world']
                t_base2board = calibration_handeye_result['t_base2world']
            else:
                if os.path.exists(f'{self.calibrate_result_dir}/calibration_handeye_result.pkl'):
                    with open(f'{self.calibrate_result_dir}/calibration_handeye_result.pkl', 'rb') as f:
                        calibration_handeye_result = pickle.load(f)
                    R_base2board = calibration_handeye_result['R_base2world']
                    t_base2board = calibration_handeye_result['t_base2world']
                else:
                    R_base2board = None
                    t_base2board = None
            rvecs, tvecs = self.fixed_camera_calibrate()
            print('calibration finished')
        else:
            with open(f'{self.calibrate_result_dir}/calibration_handeye_result.pkl', 'rb') as f:
                calibration_handeye_result = pickle.load(f)
            with open(f'{self.calibrate_result_dir}/rvecs.pkl', 'rb') as f:
                rvecs = pickle.load(f)
            with open(f'{self.calibrate_result_dir}/tvecs.pkl', 'rb') as f:
                tvecs = pickle.load(f)
            R_base2board = calibration_handeye_result['R_base2world']
            t_base2board = calibration_handeye_result['t_base2world']

        self.R_cam2world = {}
        self.t_cam2world = {}
        self.R_base2world = R_base2board
        self.t_base2world = t_base2board

        for i in range(self.n_fixed_cameras):
            device = self.serial_numbers[i]
            R_world2cam = cv2.Rodrigues(rvecs[device])[0]
            t_world2cam = tvecs[device][:, 0]
            self.R_cam2world[device] = R_world2cam.T
            self.t_cam2world[device] = -R_world2cam.T @ t_world2cam

    def get_pusher_points(self):
        assert self.R_base2world is not None
        assert self.t_base2world is not None
        R_gripper2base, t_gripper2base = self.get_robot_pose()
        R_gripper2world = self.R_base2world @ R_gripper2base
        t_gripper2world = self.R_base2world @ t_gripper2base + self.t_base2world
        pusher_points_in_world = R_gripper2world @ self.pusher_points.T + t_gripper2world[:, np.newaxis]
        pusher_points_in_world = pusher_points_in_world.T
        return pusher_points_in_world
