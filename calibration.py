import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

# 图像路径
image_files = glob.glob(r'E:/calibrationtest/pic/*.bmp')

# 棋盘格尺寸 (已知每格的大小为5mm)
square_size = 5.0  # 单位：毫米
chessboard_size = (16, 33)

# 3D 世界坐标系中的点 (0,0,0), (1,0,0), ..., (8,5,0)
obj_points = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
obj_points[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
obj_points = obj_points * square_size  # 转换为毫米单位

# 存储图像中的点和世界坐标系中的点
image_points = []
object_points = []

# 遍历图像并检测角点
for image_path in image_files:
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图像: {image_path}")
        continue  # 跳过无法读取的图像

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 查找棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        object_points.append(obj_points)
        image_points.append(corners)
        # 显示角点
        cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
        cv2.imshow('Chessboard', img)
        cv2.waitKey(500)
    else:
        print(f"未能在图像 {image_path} 中找到角点")

cv2.destroyAllWindows()

# 相机标定
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, gray.shape[::-1], None, None)

# 打印相机内参矩阵和畸变系数
print("相机内参矩阵：\n", mtx)
print("畸变系数：\n", dist)

# 校正图像
img = cv2.imread(image_files[0])
undistorted_img = cv2.undistort(img, mtx, dist)

# 显示校正后的图像
plt.imshow(cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2RGB))
plt.show()

# 重投影误差
total_error = 0
for i in range(len(object_points)):
    img_points2, _ = cv2.projectPoints(object_points[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(image_points[i], img_points2, cv2.NORM_L2) / len(img_points2[0])
    total_error += error
print(f"总重投影误差: {total_error / len(object_points)}")
