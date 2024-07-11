import numpy as np
import open3d as o3d


def rpy_to_rotation_matrix(roll, pitch, yaw):
    # Assume the input in in degree
    roll = roll / 180 * np.pi
    pitch = pitch / 180 * np.pi
    yaw = yaw / 180 * np.pi
    # Define the rotation matrices
    Rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    # Combine the rotations
    R = Rz @ Ry @ Rx
    return R


def rotation_matrix_to_rpy(rotation_matrix):
    # Get the roll, pitch, yaw in radian
    roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
    pitch = np.arctan2(-rotation_matrix[2, 0], np.sqrt(rotation_matrix[2, 1] ** 2 + rotation_matrix[2, 2] ** 2))
    yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    # Get the roll, pitch, yaw in degree
    roll = roll / np.pi * 180
    pitch = pitch / np.pi * 180
    yaw = yaw / np.pi * 180
    return roll, pitch, yaw


def depth2fgpcd(depth, intrinsic_matrix):
    H, W = depth.shape
    x, y = np.meshgrid(np.arange(W), np.arange(H))
    x = x.reshape(-1)
    y = y.reshape(-1)
    depth = depth.reshape(-1)
    points = np.stack([x, y, np.ones_like(x)], axis=1)
    points = points * depth[:, None]
    points = points @ np.linalg.inv(intrinsic_matrix).T
    return points


def similarity_transform(from_points, to_points):
    
    assert len(from_points.shape) == 2, \
        "from_points must be a m x n array"
    assert from_points.shape == to_points.shape, \
        "from_points and to_points must have the same shape"
    
    N, m = from_points.shape
    
    mean_from = from_points.mean(axis = 0)
    mean_to = to_points.mean(axis = 0)
    
    delta_from = from_points - mean_from # N x m
    delta_to = to_points - mean_to       # N x m
    
    sigma_from = (delta_from * delta_from).sum(axis = 1).mean()
    sigma_to = (delta_to * delta_to).sum(axis = 1).mean()
    
    cov_matrix = delta_to.T.dot(delta_from) / N
    
    U, d, V_t = np.linalg.svd(cov_matrix, full_matrices = True)
    cov_rank = np.linalg.matrix_rank(cov_matrix)
    S = np.eye(m)
    
    if cov_rank >= m - 1:
        S[m-1, m-1] = -1
    elif cov_rank < m-1:
        raise ValueError("colinearility detected in covariance matrix:\n{}".format(cov_matrix))
    
    R = U.dot(S).dot(V_t)
    c = (d * S.diagonal()).sum() / sigma_from
    t = mean_to - c*R.dot(mean_from)
    
    return c, R, t


def visualize_o3d(geometries):
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    geometries.append(frame)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector()
    viewer = o3d.visualization.Visualizer()
    viewer.create_window()
    for geometry in geometries:
        viewer.add_geometry(geometry)
    opt = viewer.get_render_option()
    # opt.show_coordinate_frame = True
    opt.background_color = np.asarray([1., 1., 1.])
    viewer.run()
    # viewer.destroy_window()