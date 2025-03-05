import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from matplotlib import font_manager

font_path = 'C:/Windows/Fonts/simhei.ttf'
font_prop = font_manager.FontProperties(fname=font_path)
def detect_chessboard_corners(gray, chessboard_size):
    """检测棋盘格角点。"""
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    if ret:
        return corners, ret
    else:
        return None, ret

def calibrate_camera(image_files, chessboard_size, square_size, output_dir="calibration_data", outlier_threshold=1):
    """相机标定，包含异常值剔除。"""
    obj_points = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    obj_points[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    obj_points *= square_size

    image_points = []
    object_points = []
    good_indices = [] # 存储有效图像的索引

    for i, image_path in enumerate(image_files):
        img = cv2.imread(image_path)
        if img is None:
            print(f"无法读取图像: {image_path}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ret = detect_chessboard_corners(gray, chessboard_size)

        if ret:
            object_points.append(obj_points)
            image_points.append(corners)
            good_indices.append(i)
        else:
            print(f"未能在图像 {image_path} 中找到角点")

    if len(object_points) == 0:
        print("没有检测到任何角点，标定失败。请检查图片。")
        return None, None, None, None, None

    # 标定
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, gray.shape[::-1], None, None)

    if not ret:
        print("相机标定失败，请检查图像质量或标定板配置。")
        return None, None, None, None, None

    # 重投影误差计算和异常值剔除
    total_error = 0
    errors_per_image = []
    good_object_points = []
    good_image_points = []
    good_rvecs = []
    good_tvecs = []

    for i in range(len(object_points)):
        img_points2, _ = cv2.projectPoints(object_points[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(image_points[i], img_points2, cv2.NORM_L2) / len(img_points2)
        errors_per_image.append(error)
        if error < outlier_threshold: #只保留误差较小的图像
            total_error+=error
            good_object_points.append(object_points[i])
            good_image_points.append(image_points[i])
            good_rvecs.append(rvecs[i])
            good_tvecs.append(tvecs[i])
        else:
            print(f"发现异常值图片{image_files[good_indices[i]]},误差为{error:.4f}，已剔除")

    if len(good_object_points) < len(object_points):
        print("进行第二次标定...")
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(good_object_points, good_image_points, gray.shape[::-1], None, None)

    avg_error = total_error / len(good_object_points)
    print(f"总重投影误差: {avg_error:.4f}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    np.save(os.path.join(output_dir, 'mtx.npy'), mtx)
    np.save(os.path.join(output_dir, 'dist.npy'), dist)
    np.save(os.path.join(output_dir, 'avg_error.npy'), avg_error)
    np.save(os.path.join(output_dir, 'errors_per_image.npy'), np.array(errors_per_image))
    if ret:
        np.save(os.path.join(output_dir, 'rvecs.npy'), np.array(rvecs))
        np.save(os.path.join(output_dir, 'tvecs.npy'), np.array(tvecs))
        np.save(os.path.join(output_dir, 'good_indices.npy'), np.array(good_indices))
    return mtx, dist, rvecs, tvecs, avg_error, errors_per_image, good_indices

def undistort_image(img, mtx, dist):
    """校正图像畸变。"""
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h)) #alpha参数设为1会保留所有原始图像像素，设为0则会裁剪掉黑边
    undistorted_img = cv2.undistort(img, mtx, dist, None, newcameramtx)
    x, y, w, h = roi
    undistorted_img = undistorted_img[y:y+h, x:x+w]
    return undistorted_img

# 加载支持中文的字体
font_path = 'C:/Windows/Fonts/simhei.ttf'  # 替换为你的字体路径
font_prop = font_manager.FontProperties(fname=font_path)

# 配置文件
config = {
    "image_dir": "E:/calibrationtest/pic",
    "chessboard_size": (16, 33),
    "square_size": 5.0,
    "output_dir": "calibration_data",
    "outlier_threshold":0.012 #重投影误差阈值，超过这个值就认为是异常值并剔除
}

image_files = glob.glob(config['image_dir'] + '/*.bmp')

if not image_files:
    print("没有找到任何图片，请检查路径。")
else:
    mtx, dist, rvecs, tvecs, avg_error, errors_per_image, good_indices = calibrate_camera(image_files, config['chessboard_size'], config['square_size'], config["output_dir"], config["outlier_threshold"])
    if mtx is not None:
        print("相机内参矩阵：\n", mtx)
        print("畸变系数：\n", dist)
        print(f"总重投影误差为{avg_error:.4f}")
        print("以下是每张图片的重投影误差:")
        for i in range(len(errors_per_image)):
            print(f"第{i+1}张: {errors_per_image[i]:.4f}")
        if len(good_indices)<len(image_files):
            print(f"有效图片为{len(good_indices)}/{len(image_files)}")
            print(f"有效图片的索引为{good_indices}")

        # 可视化每张图像的重投影误差
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(errors_per_image) + 1), errors_per_image, marker='o', linestyle='-', color='b')
        plt.axhline(y=config["outlier_threshold"], color='r', linestyle='--',
                    label=f'异常值阈值 ({config["outlier_threshold"]})')
        plt.xlabel("图像编号", fontproperties=font_prop)
        plt.ylabel("重投影误差", fontproperties=font_prop)
        plt.title("每张图像的重投影误差", fontproperties=font_prop)
        plt.legend(prop=font_prop)
        plt.grid(True)
        plt.show()

        # 校正图像（所有有效图片）
        if good_indices:
            for i, index in enumerate(good_indices):
                img = cv2.imread(image_files[index])
                undistorted_img = undistort_image(img, mtx, dist)

                # 创建对比图
                fig, axs = plt.subplots(1, 2, figsize=(10, 5))
                axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                axs[0].set_title("原始图像", fontproperties=font_prop)
                axs[0].axis("off")

                axs[1].imshow(cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2RGB))
                axs[1].set_title("校正后的图像", fontproperties=font_prop)
                axs[1].axis("off")

                plt.suptitle(f"第{i + 1}张图像矫正效果", fontproperties=font_prop)
                plt.tight_layout()
                plt.show()

    else:
        print("相机标定失败。")