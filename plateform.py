import numpy as np
import cv2
import matplotlib.pyplot as plt
from screeninfo import get_monitors

# 加载标定数据
calibration_data_path = "calibration_data/"
mtx = np.load(calibration_data_path + "camera_matrix.npy")
dist = np.load(calibration_data_path + "dist_coefficients.npy")
rotation_matrices = np.load(calibration_data_path + "rotation_matrices.npy")
translation_vectors = np.load(calibration_data_path + "translation_vectors.npy")

print("相机内参矩阵：", mtx)
print("畸变系数：", dist)

# 棋盘格尺寸和每格大小
square_size = 5.0  # 每格大小，单位：毫米
chessboard_size = (17, 34)  # 17x34 的棋盘格，实际角点数量为 16x33

# 读取标定板图像
board_image_path = "E:/calibrationtest/pic/1846-00.bmp"  # 你的标定板图像路径
board_img = cv2.imread(board_image_path)

# 畸变矫正：使用相机内参和畸变系数
undistorted_img = cv2.undistort(board_img, mtx, dist)

# 检测棋盘格角点
ret, corners = cv2.findChessboardCorners(undistorted_img,
                                         (chessboard_size[0] - 1, chessboard_size[1] - 1))  # 内部角点数量为 (15, 32)
if ret:
    print(f"成功检测到角点：{len(corners)}个")

    # 将角点按列存储 (X, Y)
    sorted_corners = corners.reshape(-1, 2)  # 将角点平展为一个Nx2的数组

    # 获取图像尺寸
    img_height, img_width = undistorted_img.shape[:2]

    # 按列拟合直线并连接角点
    for col in range(chessboard_size[0] - 1):  # 遍历每一列
        column_points = sorted_corners[col::(chessboard_size[0] - 1)]  # 获取该列所有角点
        if len(column_points) >= 2:  # 至少需要两个点来拟合一条直线
            # 使用cv2的fitLine拟合直线，返回参数为 (vx, vy, x0, y0)
            [vx, vy, x0, y0] = cv2.fitLine(np.array(column_points, dtype=np.float32), cv2.DIST_L2, 0, 0.01, 0.01)

            # 提取方向向量的x和y分量作为标量
            vx = float(vx[0])  # 将 vx 作为标量
            vy = float(vy[0])  # 将 vy 作为标量

            # 提取直线上的点（x0, y0）作为标量
            x0 = float(x0[0])  # 将 x0 作为标量
            y0 = float(y0[0])  # 将 y0 作为标量

            # 计算直线与图像边缘的交点
            slope = vy / vx  # 计算直线的斜率
            intercept = y0 - slope * x0  # 计算直线的截距

            # 计算与左边界 (x=0) 的交点
            y1 = int(slope * 0 + intercept)
            # 计算与右边界 (x=img_width-1) 的交点
            y2 = int(slope * (img_width - 1) + intercept)

            # 计算与上边界 (y=0) 的交点
            x3 = int((0 - intercept) / slope)
            # 计算与下边界 (y=img_height-1) 的交点
            x4 = int((img_height - 1 - intercept) / slope)


            pt1 = (x3, 0)

            pt2 = (x4, img_height - 1)

            # 画出直线
            cv2.line(undistorted_img, pt1, pt2, (0, 255, 0), 1)  # 画出直线


    # 按行拟合直线并连接角点
    for row in range(chessboard_size[1] - 1):  # 遍历每一行
        row_points = sorted_corners[row * (chessboard_size[0] - 1):(row + 1) * (chessboard_size[0] - 1)]  # 获取该行所有角点
        if len(row_points) >= 2:  # 至少需要两个点来拟合一条直线
            # 使用cv2的fitLine拟合直线，返回参数为 (vx, vy, x0, y0)
            [vx, vy, x0, y0] = cv2.fitLine(np.array(row_points, dtype=np.float32), cv2.DIST_L2, 0, 0.01, 0.01)

            # 提取方向向量的x和y分量作为标量
            vx = float(vx[0])  # 将 vx 作为标量
            vy = float(vy[0])  # 将 vy 作为标量

            # 提取直线上的点（x0, y0）作为标量
            x0 = float(x0[0])  # 将 x0 作为标量
            y0 = float(y0[0])  # 将 y0 作为标量

            # 计算直线与图像边缘的交点
            slope = vy / vx
            intercept = y0 - slope * x0

            # 计算与上边界 (y=0) 的交点
            x1 = int((0 - intercept) / slope)
            # 计算与下边界 (y=img_height-1) 的交点
            x2 = int((img_height - 1 - intercept) / slope)

            # 计算与左边界 (x=0) 的交点
            y3 = int(slope * 0 + intercept)
            # 计算与右边界 (x=img_width-1) 的交点
            y4 = int(slope * (img_width - 1) + intercept)

            pt3 = (x1, 0)

            pt4 = (x2, img_height - 1)

            cv2.line(undistorted_img, pt3, pt4, (0, 255, 255), 1)  # 用黄色绘制延长后的直线

    # 显示原始畸变图像，已绘制延长的拟合直线和角点连接
    cv2.imshow("Original Distorted Image with Extended Fitted Lines", undistorted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    print("未能检测到角点，检查图像和棋盘格尺寸")
