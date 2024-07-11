import open3d as o3d
import os
import time
import numpy as np
import cv2

des_dir = 'test_GranularManip'
cam_idx_lst = [0, 4, 8, 13]
pcd_lst = []
vis = o3d.visualization.Visualizer()
vis.create_window()
for i, cam_idx in enumerate(cam_idx_lst):
    pcd = o3d.io.read_point_cloud(os.path.join(des_dir, 'pcd_0_cam_%d.ply' % cam_idx))
    pcd_lst.append(pcd)
    vis.add_geometry(pcd_lst[i])

mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
vis.add_geometry(mesh)

view_control = vis.get_view_control()
cam_model = view_control.convert_to_pinhole_camera_parameters()

deg = np.arange(0, 2*np.pi, 2*np.pi/120.)
dis = 7.
h = 4.
theta = np.arctan(h / dis)
out = cv2.VideoWriter(os.path.join(des_dir, 'pc_stitch.avi'),cv2.VideoWriter_fourcc(*'DIVX'), 10, (1853, 1025))
for i in range(120):
    # cam_model.extrinsic = np.array([[np.sin(deg[i]), -np.sin(theta)*np.cos(deg[i]), np.cos(theta)*np.cos(deg[i]), -dis*np.cos(deg[i])],
    #                                 [-np.cos(deg[i]), -np.sin(theta)*np.sin(deg[i]), np.cos(theta)*np.sin(deg[i]), -dis*np.sin(deg[i])],
    #                                 [0, np.cos(theta), np.sin(theta), h],
    #                                 [0, 0, 0, 1]])
    # cam_model.extrinsic = np.array([[1, 0, 0, -dis],
    #                                 [0, 1, 0, -h],
    #                                 [0, 0, 1, 0],
    #                                 [0, 0, 0, 1]])
    # vis.set_view_control_parameters(cam_model)
    # view_control.convert_from_pinhole_camera_parameters(cam_model)
    view_control.set_front([np.cos(theta)*np.sin(deg[i]), np.sin(theta), np.cos(theta)*np.cos(deg[i])])
    view_control.set_up([-np.sin(theta)*np.sin(deg[i]), np.sin(theta), -np.sin(theta)*np.cos(deg[i])])
    view_control.set_lookat([0, 0, 0])
    for pcd_i, cam_idx in enumerate(cam_idx_lst):
        pcd_new = o3d.io.read_point_cloud(os.path.join(des_dir, 'pcd_%d_cam_%d.ply' % (i, cam_idx)))
        pcd_lst[pcd_i].points = pcd_new.points
        pcd_lst[pcd_i].colors = pcd_new.colors
        vis.update_geometry(pcd_lst[pcd_i])
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(os.path.join(des_dir, 'img_%d.png' % i))
    img = cv2.imread(os.path.join(des_dir, 'img_%d.png' % i))
    out.write(img)

    time.sleep(0.05)
    vis.run()

out.release()
vis.destroy_window()