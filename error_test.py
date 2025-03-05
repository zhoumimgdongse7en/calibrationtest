import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

# 图像路径
image_files = glob.glob(r'E:/calibrationtest/pic/*.bmp')

# 棋盘格尺寸
square_size = 5.0  # 单位：毫米
chessboard_size = (16, 33)

# 3D 世界坐标系中的点
obj_points = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
obj_points[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
obj_points = obj_points * square_size  # 转换为毫米单位

# 存储图像中的点和世界坐标系中的点
image_points = []
object_points = []
image_positions = []  # 存储图像的相对位置编号

# 遍历图像并检测角点
for image_path in image_files:
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图像: {image_path}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 查找棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        object_points.append(obj_points)
        image_points.append(corners)

        # 假设文件名格式如"1846-4", 从中提取相对位置
        position = int(image_path.split('-')[-1].split('.')[0])  # 提取数字部分
        image_positions.append(position)

        # 显示角点
        cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
        cv2.imshow('Chessboard', img)
        cv2.waitKey(500)
    else:
        print(f"未能在图像 {image_path} 中找到角点")

cv2.destroyAllWindows()

# 相机标定
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, gray.shape[::-1], None, None)

# 计算每个标定板之间的相对距离
for i in range(1, len(rvecs)):
    # 获取标定板的平移向量 (tvecs是平移向量，表示标定板在世界坐标系中的位置)
    estimated_position_1 = tvecs[i - 1]
    estimated_position_2 = tvecs[i]

    # 计算两个标定板之间的实际相对位置
    actual_distance = abs(image_positions[i] - image_positions[i - 1])  # 期望的相对位置
    # 计算估计的平移向量之间的欧几里得距离
    estimated_distance = np.linalg.norm(estimated_position_2 - estimated_position_1)

    # 计算误差
    distance_error = abs(estimated_distance - actual_distance)

    print(
        f"标定板 {i} 和标定板 {i - 1} 的估计距离: {estimated_distance:.2f} mm, 实际距离: {actual_distance} mm, 误差: {distance_error:.2f} mm")

# 显示校正后的图像
img = cv2.imread(image_files[0])
undistorted_img = cv2.undistort(img, mtx, dist)

plt.imshow(cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2RGB))
plt.show()
