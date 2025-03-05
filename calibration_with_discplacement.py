import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import scipy.optimize

# 图像路径
image_files = glob.glob(r'E:/calibrationtest/pic/*.bmp')

# 棋盘格尺寸 (已知每格的大小为5mm)
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


# 定义优化目标函数
def optimization_function(params, object_points, image_points, camera_matrix, dist_coeffs, image_positions):
    # 提取优化参数
    rvecs = params[:len(object_points) * 3].reshape(-1, 3)
    tvecs = params[len(object_points) * 3:].reshape(-1, 3)

    # 计算重投影误差
    total_error = 0
    for i in range(len(object_points)):
        img_points2, _ = cv2.projectPoints(object_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv2.norm(image_points[i], img_points2, cv2.NORM_L2) / len(img_points2)
        total_error += error

    # 加入相对位置约束
    relative_position_error = 0
    for i in range(1, len(rvecs)):
        expected_distance = image_positions[i] - image_positions[i - 1]  # 期望的相对位置
        rel_position = np.linalg.norm(tvecs[i] - tvecs[i - 1]) - expected_distance  # 计算实际距离与期望距离的差异
        relative_position_error += rel_position ** 2  # 累加误差的平方

    return total_error + relative_position_error


# 初始标定
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, gray.shape[::-1], None, None)
# 确保 rvecs 和 tvecs 被正确初始化
if len(rvecs) == 0 or len(tvecs) == 0:
    raise ValueError("未能从标定中获取旋转向量或平移向量")
# 展开所有的 rvecs 和 tvecs 用于优化
initial_params = np.hstack([rvec.flatten() for rvec in rvecs] + [tvec.flatten() for tvec in tvecs])


# 使用优化算法
result = scipy.optimize.least_squares(optimization_function, initial_params,
                                      args=(object_points, image_points, mtx, dist, image_positions))

# 得到优化后的参数
optimized_rvecs = result.x[:len(object_points) * 3].reshape(-1, 3)
optimized_tvecs = result.x[len(object_points) * 3:].reshape(-1, 3)

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
    img_points2, _ = cv2.projectPoints(object_points[i], optimized_rvecs[i], optimized_tvecs[i], mtx, dist)
    error = cv2.norm(image_points[i], img_points2, cv2.NORM_L2) / len(img_points2)
    total_error += error
print(f"总重投影误差: {total_error / len(object_points)}")
