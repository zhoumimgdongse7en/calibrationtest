import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os
from matplotlib import cm
from matplotlib.colors import Normalize
import json

def load_calibration_data(output_dir):
    """从保存的 .npy 文件中加载相机标定数据"""
    mtx = np.load(os.path.join(output_dir, "camera_matrix.npy"))
    dist = np.load(os.path.join(output_dir, "dist_coefficients.npy"))
    rotation_matrices = np.load(os.path.join(output_dir, "rotation_matrices.npy"))
    translation_vectors = np.load(os.path.join(output_dir, "translation_vectors.npy"))
    return mtx, dist, rotation_matrices, translation_vectors


def load_corners_from_json(corners_json_dir):
    """从 JSON 文件中加载角点数据"""
    corners_data = {}
    for filename in os.listdir(corners_json_dir):
        if filename.endswith("_corners.json"):  # 只加载角点数据文件
            file_path = os.path.join(corners_json_dir, filename)
            with open(file_path, 'r') as f:
                corners_data[filename] = json.load(f)
    return corners_data


def calculate_fov(mtx, image_width, image_height):
    """根据相机内参计算视场角"""
    fx = mtx[0, 0]  # x方向的焦距
    fy = mtx[1, 1]  # y方向的焦距

    fov_x = 2 * np.arctan(image_width / (2 * fx))  # 水平视场角，单位：弧度
    fov_y = 2 * np.arctan(image_height / (2 * fy))  # 垂直视场角，单位：弧度

    return fov_x, fov_y


def draw_fov_cone(ax, camera_position, fov_x, fov_y, far_distance=500, color='gray'):
    """绘制相机的完整视场锥体"""
    # FOV锥体的角度（弧度）
    angle_x = np.tan(fov_x / 2)
    angle_y = np.tan(fov_y / 2)

    # 在相机坐标系下定义视场锥体的四个顶点（远端）
    cone_points_camera = np.array([
        [far_distance * angle_x, far_distance * angle_y, far_distance],  # 右上角
        [-far_distance * angle_x, far_distance * angle_y, far_distance],  # 左上角
        [-far_distance * angle_x, -far_distance * angle_y, far_distance],  # 左下角
        [far_distance * angle_x, -far_distance * angle_y, far_distance],  # 右下角
    ])

    # 绘制锥体的四条边（从相机位置到远端的顶点）
    for point in cone_points_camera:
        ax.plot(
            [camera_position[0], point[0]],
            [camera_position[1], point[1]],
            [camera_position[2], point[2]],
            color=color
        )

    # 绘制锥体的底面（连接四个顶点）
    cone_points_camera = np.vstack((cone_points_camera, cone_points_camera[0]))  # 闭合
    ax.plot(
        cone_points_camera[:, 0],
        cone_points_camera[:, 1],
        cone_points_camera[:, 2],
        color=color
    )


def draw_chessboard_plane_with_points(ax, rmat, tvec, chessboard_size, square_size, image_points, mtx, color_map, norm):
    """绘制标定板的平面并在其上显示棋盘格角点"""
    rows, cols = chessboard_size

    # 标定板的四个角点（局部坐标系）
    corners = np.array([
        [0, 0, 0],  # 左下角
        [(cols - 1) * square_size, 0, 0],  # 右下角
        [(cols - 1) * square_size, (rows - 1) * square_size, 0],  # 右上角
        [0, (rows - 1) * square_size, 0]  # 左上角
    ])

    # 将角点从局部坐标系转换到相机坐标系
    corners_camera = (rmat @ corners.T).T + tvec.reshape(1, -1)

    # 计算颜色
    color = color_map(norm(np.linalg.norm(tvec)))

    # 绘制标定板平面
    verts = [list(corners_camera)]
    ax.add_collection3d(Poly3DCollection(verts, color=color, alpha=0.8))

    # 如果提供了角点数据（图像空间的角点），则反投影到3D空间并绘制
    if image_points:
        for pt in image_points:
            # 将图像坐标反投影到3D空间
            pt_homogeneous = np.array([pt[0], pt[1], 1]).reshape(3, 1)  # 图像坐标转齐次坐标
            pt_undistorted = cv2.undistortPoints(pt_homogeneous.T, mtx, dist=None).reshape(1, 2)
            pt_3d_homogeneous = np.linalg.inv(mtx) @ pt_undistorted.T  # 反投影到相机坐标系
            pt_3d = np.array([rmat @ pt_3d_homogeneous + tvec]).reshape(3, )

            # 在3D标定板平面上绘制角点
            ax.scatter(pt_3d[0], pt_3d[1], pt_3d[2], color='g', s=50)


def visualize_camera_and_chessboard_with_points(rotation_matrices, translation_vectors, chessboard_size, square_size, mtx, image_width, image_height, image_points):
    """可视化相机和标定板的位置关系并显示标定点"""
    # 相机的默认位置和旋转（Z 轴正方向）
    camera_position = np.array([0, 0, 0])

    # 计算视场角
    fov_x, fov_y = calculate_fov(mtx, image_width, image_height)
    print(f"水平视场角: {np.degrees(fov_x):.2f}°")
    print(f"垂直视场角: {np.degrees(fov_y):.2f}°")

    # 创建 3D 图形
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # 绘制相机的位置
    ax.scatter(*camera_position, c="r", s=100, label="Camera")

    # 绘制视场锥体
    draw_fov_cone(ax, camera_position, fov_x, fov_y)

    # 设置颜色映射
    color_map = cm.viridis
    distances = []
    for tvec in translation_vectors:
        # 计算标定板中心点到相机的距离
        center_point = tvec.reshape(-1)
        distance = np.linalg.norm(center_point - camera_position)
        distances.append(distance)

    # 根据距离设置归一化范围
    norm = Normalize(vmin=min(distances), vmax=max(distances))

    # 绘制标定板，并标记交点
    for i, (rmat, tvec) in enumerate(zip(rotation_matrices, translation_vectors)):
        # 获取当前标定板的角点
        image_points_for_board = image_points.get(f'corners.json', [])
        # 绘制标定板平面，并在上面标记交点
        draw_chessboard_plane_with_points(ax, rmat, tvec, chessboard_size, square_size, image_points_for_board, mtx, color_map, norm)

    # 设置轴标签和范围
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim([-200, 200])
    ax.set_ylim([-200, 200])
    ax.set_zlim([0, 500])
    ax.legend()

    # 显示图形
    plt.show()


# 配置文件
config = {
    "output_dir": "calibration_data",  # 标定数据路径
    "chessboard_size": (16, 33),  # 棋盘格的内角点数
    "square_size": 5.0,  # 格子大小，单位毫米
    "corners_json_dir": "E:/calibrationtest/results"  # 角点 JSON 文件路径
}

# 加载标定数据
mtx, dist, rotation_matrices, translation_vectors = load_calibration_data(config['output_dir'])

# 加载角点数据
image_points = load_corners_from_json(config['corners_json_dir'])

# 图像大小，假设每张图像的分辨率
image_width, image_height = 1600, 1200

# 可视化
visualize_camera_and_chessboard_with_points(
    rotation_matrices, translation_vectors, config["chessboard_size"], config["square_size"], mtx, image_width, image_height, image_points
)
