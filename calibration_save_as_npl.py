import cv2
import numpy as np
import glob
import os
import json

def detect_chessboard_corners(gray, chessboard_size):
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    return corners, ret


def save_calibration_to_npy(output_dir, mtx, dist, avg_error, errors_per_image, rvecs, tvecs, depths):
    """保存标定结果和深度信息到 .npy 文件"""
    # 保存时，确保每个旋转向量都转化为旋转矩阵
    rotation_matrices = np.array([cv2.Rodrigues(rvec)[0] for rvec in rvecs])  # 旋转向量转旋转矩阵
    translation_vectors = np.array(tvecs)  # 平移向量

    # 确保保存为正确的形状
    np.save(os.path.join(output_dir, "camera_matrix.npy"), mtx)  # 保存相机矩阵
    np.save(os.path.join(output_dir, "dist_coefficients.npy"), dist)  # 保存畸变系数
    np.save(os.path.join(output_dir, "rotation_matrices.npy"), rotation_matrices)  # 保存旋转矩阵
    np.save(os.path.join(output_dir, "translation_vectors.npy"), translation_vectors)  # 保存平移向量
    np.save(os.path.join(output_dir, "depths.npy"), depths)  # 保存深度信息

def undistort_image(img, mtx, dist):
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    undistorted_img = cv2.undistort(img, mtx, dist, None, newcameramtx)
    x, y, w, h = roi
    return undistorted_img[y:y+h, x:x+w]

def calculate_depths(object_points, image_points, mtx, dist, rvecs, tvecs):
    """通过投影矩阵计算每个角点的深度"""
    depths = []
    for i in range(len(object_points)):
        # 使用相机内外参数，将3D世界坐标投影到图像坐标
        img_points, _ = cv2.projectPoints(object_points[i], rvecs[i], tvecs[i], mtx, dist)
        # 假设Z为棋盘格角点的深度（在平面上可以假设为Z=0）
        X, Y, Z = object_points[i][0], object_points[i][1], object_points[i][2]
        depth = Z  # 深度即为世界坐标中的Z值
        depths.append(depth)
    return depths

def calibrate_camera(image_files, chessboard_size, square_size, output_dir, outlier_threshold):
    """相机标定，并计算每个角点的深度"""
    obj_points = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    obj_points[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

    object_points, image_points = [], []
    for image_path in image_files:
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ret = detect_chessboard_corners(gray, chessboard_size)
        if ret:
            object_points.append(obj_points)
            image_points.append(corners)

    if not object_points:
        print("没有检测到有效的角点。")
        return None, None, None, None, None, None

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, gray.shape[::-1], None, None)
    if not ret:
        print("相机标定失败。")
        return None, None, None, None, None, None

    total_error = 0
    errors_per_image = []
    for i in range(len(object_points)):
        img_points2, _ = cv2.projectPoints(object_points[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(image_points[i], img_points2, cv2.NORM_L2) / len(img_points2)
        errors_per_image.append(error)
        total_error += error

    avg_error = total_error / len(object_points)

    # 计算每个角点的深度信息
    depths = calculate_depths(object_points, image_points, mtx, dist, rvecs, tvecs)

    # 保存标定结果和深度信息
    save_calibration_to_npy(output_dir, mtx, dist, avg_error, errors_per_image, rvecs, tvecs, depths)

    return mtx, dist, avg_error, errors_per_image, rvecs, tvecs


# 配置文件
config = {
    "image_dir": "E:/calibrationtest/pic",  # 输入文件夹路径，替换为实际路径
    "chessboard_size": (16, 33),  # 棋盘格的内角点数，替换为实际大小
    "square_size": 5.0,  # 格子大小，单位为毫米
    "output_dir": "calibration_data",  # 输出路径，替换为实际路径
    "outlier_threshold": 0.012,  # 畸变参数阈值
}

image_files = glob.glob(config["image_dir"] + "/*.bmp")
if image_files:
    mtx, dist, avg_error, errors_per_image, rvecs, tvecs = calibrate_camera(
        image_files, config["chessboard_size"], config["square_size"], config["output_dir"], config["outlier_threshold"]
    )
    print(f"标定完成，平均重投影误差: {avg_error:.4f}")
else:
    print("未找到标定图片，请检查路径。")
