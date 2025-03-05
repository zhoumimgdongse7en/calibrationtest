import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
import os

def save_chessboard_lines(filename, lines):
    """
    保存棋盘格线到文件中。
    :param filename: 保存的文件名
    :param lines: 棋盘格线参数列表，每条线是一个字典 {vx, vy, x0, y0}
    """
    with open(filename, 'w') as f:
        json.dump(lines, f, indent=4)
    print(f"棋盘格线已保存到 {filename}")

def save_chessboard_corners(filename, corners):
    """
    保存棋盘格角点图像坐标到文件中。
    :param filename: 保存的文件名
    :param corners: 棋盘格角点图像坐标（Nx2 数组）
    """
    with open(filename, 'w') as f:
        json.dump(corners.tolist(), f, indent=4)  # 将角点坐标转换为列表
    print(f"棋盘格角点已保存到 {filename}")

def load_chessboard_lines(filename):
    """
    从文件加载棋盘格线。
    :param filename: 保存的文件名
    :return: 棋盘格线参数列表
    """
    with open(filename, 'r') as f:
        lines = json.load(f)
    print(f"棋盘格线已从 {filename} 加载")
    return lines

def process_chessboard_image(board_image_path, save_directory, calibration_data):
    """
    对单张棋盘格图像进行处理，包括畸变校正、棋盘格线拟合和保存结果。
    """
    mtx, dist = calibration_data

    # 读取图像
    board_img = cv2.imread(board_image_path)
    if board_img is None:
        print(f"无法读取图像 {board_image_path}")
        return

    # 畸变矫正：使用相机内参和畸变系数
    undistorted_img = cv2.undistort(board_img, mtx, dist)

    # 检测棋盘格角点
    chessboard_size = (17, 34)  # 17x34 的棋盘格，实际角点数量为 (16, 33)
    ret, corners = cv2.findChessboardCorners(undistorted_img,
                                             (chessboard_size[0] - 1, chessboard_size[1] - 1))
    if not ret:
        print(f"未能检测到棋盘格角点，跳过图像 {board_image_path}")
        return

    print(f"成功检测到图像 {board_image_path} 的角点：{len(corners)} 个")

    # 将角点按列存储 (X, Y)
    sorted_corners = corners.reshape(-1, 2)  # 将角点平展为一个 Nx2 的数组

    # 获取图像尺寸
    img_height, img_width = undistorted_img.shape[:2]

    # 存储棋盘格线的参数
    chessboard_lines = []

    # 按列拟合直线并连接角点
    for col in range(chessboard_size[0] - 1):  # 遍历每一列
        column_points = sorted_corners[col::(chessboard_size[0] - 1)]  # 获取该列所有角点
        if len(column_points) >= 2:  # 至少需要两个点来拟合一条直线
            [vx, vy, x0, y0] = cv2.fitLine(np.array(column_points, dtype=np.float32), cv2.DIST_L2, 0, 0.01, 0.01)
            chessboard_lines.append({"vx": float(vx[0]), "vy": float(vy[0]), "x0": float(x0[0]), "y0": float(y0[0])})

    # 按行拟合直线并连接角点
    for row in range(chessboard_size[1] - 1):  # 遍历每一行
        row_points = sorted_corners[row * (chessboard_size[0] - 1):(row + 1) * (chessboard_size[0] - 1)]  # 获取该行所有角点
        if len(row_points) >= 2:  # 至少需要两个点来拟合一条直线
            [vx, vy, x0, y0] = cv2.fitLine(np.array(row_points, dtype=np.float32), cv2.DIST_L2, 0, 0.01, 0.01)
            chessboard_lines.append({"vx": float(vx[0]), "vy": float(vy[0]), "x0": float(x0[0]), "y0": float(y0[0])})

    # 保存棋盘格线到文件
    image_name = os.path.splitext(os.path.basename(board_image_path))[0]
    save_chessboard_lines(os.path.join(save_directory, f"{image_name}_lines.json"), chessboard_lines)

    # 保存棋盘格角点图像坐标到文件
    save_chessboard_corners(os.path.join(save_directory, f"{image_name}_corners.json"), sorted_corners)

    # 可视化结果并保存
    for line in chessboard_lines:
        vx, vy, x0, y0 = line["vx"], line["vy"], line["x0"], line["y0"]
        slope = vy / vx
        intercept = y0 - slope * x0

        x1 = 0
        y1 = int(intercept)
        x2 = img_width - 1
        y2 = int(slope * x2 + intercept)

        # 绘制直线
        cv2.line(undistorted_img, (x1, y1), (x2, y2), (0, 255, 0), 1)

    # 保存可视化结果
    save_image_path = os.path.join(save_directory, f"{image_name}_result.png")
    cv2.imwrite(save_image_path, undistorted_img)
    print(f"处理结果已保存到 {save_image_path}")


def process_directory(input_directory, save_directory, calibration_data):
    """
    遍历文件夹中的所有 .bmp 文件，对每张图片进行棋盘格线拟合处理。
    """
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    for filename in os.listdir(input_directory):
        if filename.endswith(".bmp"):
            file_path = os.path.join(input_directory, filename)
            print(f"开始处理图像 {file_path}...")
            process_chessboard_image(file_path, save_directory, calibration_data)


def main():
    # 加载标定数据
    calibration_data_path = "calibration_data/"
    mtx = np.load(calibration_data_path + "camera_matrix.npy")
    dist = np.load(calibration_data_path + "dist_coefficients.npy")
    calibration_data = (mtx, dist)

    # 输入目录和输出目录
    input_directory = "E:/calibrationtest/pic"  # 棋盘格图片文件夹路径
    save_directory = "E:/calibrationtest/results"  # 保存结果的文件夹路径

    # 遍历并处理目录中的所有 .bmp 文件
    process_directory(input_directory, save_directory, calibration_data)


if __name__ == "__main__":
    main()
