import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
import matplotlib
import os
import csv
import json


# 设置支持中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
font_path = 'C:/Windows/Fonts/simhei.ttf'
font_prop = font_manager.FontProperties(fname=font_path)


def load_calibration_params(calibration_folder):
    """加载相机标定参数"""
    mtx = np.load(os.path.join(calibration_folder, 'camera_matrix.npy'))
    dist = np.load(os.path.join(calibration_folder, 'dist_coefficients.npy'))
    return mtx, dist


def adjust_exposure_and_threshold(undistorted_img, alpha, beta, threshold):
    """调整图像曝光并进行二值化处理"""
    adjusted_img = cv2.convertScaleAbs(undistorted_img, alpha=alpha, beta=beta)
    _, binary_img = cv2.threshold(adjusted_img, threshold, 255, cv2.THRESH_BINARY)
    return adjusted_img, binary_img


def select_roi(img):
    """手动框选感兴趣区域 (ROI)"""
    print("请手动框选光条所在区域，按回车确认，按 ESC 取消")
    roi = cv2.selectROI("选择光条区域", img)
    cv2.destroyAllWindows()
    x, y, w, h = map(int, roi)
    return img[y:y + h, x:x + w], x, y, w, h


def extract_light_stripe_gray_center_with_exposure(image_path, alpha, beta, threshold, kernel_size, mtx, dist):
    """使用增加曝光预处理和灰度中心法提取光条中心线"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法加载图像: {image_path}")

    # 图像畸变校正
    h, w = img.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    undistorted_img = cv2.undistort(img, mtx, dist, None, new_camera_matrix)

    # 手动框选感兴趣区域 (ROI)
    roi_img, x_offset, y_offset, _, _ = select_roi(undistorted_img)

    # 曝光调整和二值化
    adjusted_img, binary_img = adjust_exposure_and_threshold(roi_img, alpha=alpha, beta=beta, threshold=threshold)

    # 平滑图像以减少噪声
    blurred = cv2.GaussianBlur(adjusted_img, (kernel_size, kernel_size), 0)

    # 提取光条中心点
    centers = []
    output_image = cv2.cvtColor(undistorted_img, cv2.COLOR_GRAY2BGR)  # 原始图像的彩色副本，用于叠加拟合直线

    for row in range(roi_img.shape[0]):
        indices = np.where(binary_img[row, :] > 0)[0]
        if len(indices) > 0:
            weights = blurred[row, indices]
            center_x = int(np.sum(weights * indices) / np.sum(weights))
            # 映射回原始图像坐标
            original_x = center_x + x_offset
            original_y = row + y_offset
            centers.append((original_x, original_y))
            # 在原始图像上绘制中心点
            cv2.circle(output_image, (original_x, original_y), 1, (0, 0, 255), -1)

    return centers, output_image, adjusted_img, binary_img, roi_img


def save_coordinates_to_csv(centers, csv_path):
    """将提取的光条中心坐标保存到CSV文件"""
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["X", "Y"])
        for center in centers:
            writer.writerow(center)


def save_line_params_to_json(m, c, json_path, filename):
    """将拟合的光条直线参数保存到 JSON 文件"""
    data = {
        "filename": filename,
        "line_params": {
            "slope": m,
            "intercept": c
        }
    }
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"拟合直线参数已保存到 {json_path}")


def fit_lines_and_visualize(centers, original_image, save_path, save_params_path, line_only_path, json_path, filename):
    """对提取的点进行直线拟合并可视化，保存拟合参数，同时保存拟合直线单独图像和JSON文件"""
    # 将中心点分成 x 和 y 坐标
    points = np.array(centers)
    x_coords, y_coords = points[:, 0], points[:, 1]

    # 直线拟合 (y = mx + c)
    fit_params = np.polyfit(x_coords, y_coords, 1)  # 一阶多项式拟合
    m, c = fit_params

    # 计算拟合直线的两个端点
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = m * x_min + c, m * x_max + c

    # 在原始图像上绘制拟合直线
    image_with_line = original_image.copy()  # 复制原始图像用于绘制
    cv2.line(image_with_line, (x_min, int(y_min)), (x_max, int(y_max)), (255, 0, 0), 2)  # 绘制拟合直线

    # 可视化：原始图像 + 拟合直线 + 光条中心点
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image_with_line, cv2.COLOR_BGR2RGB))
    plt.scatter(x_coords, y_coords, color='red', s=1, label='提取的中心点')
    plt.plot([x_min, x_max], [y_min, y_max], color='blue', label='拟合直线', linewidth=1)
    plt.title("光条中心点与拟合直线", fontproperties=font_prop)
    plt.legend()
    plt.axis("off")

    # 保存叠加了拟合直线的图像
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 保存仅拟合直线的图像
    image_with_line_only = np.zeros_like(original_image)  # 创建一张全黑图像
    cv2.line(image_with_line_only, (x_min, int(y_min)), (x_max, int(y_max)), (255, 0, 0), 2)  # 绘制拟合直线
    cv2.imwrite(line_only_path, image_with_line_only)  # 保存仅包含拟合直线的图像

    # 保存拟合直线的参数到 JSON 文件
    save_line_params_to_json(m, c, json_path, filename)


def process_directory(input_dir, output_dir, params_dir, line_dir, json_dir, alpha=1.5, beta=50, threshold=200, kernel_size=5, calibration_folder=None):
    """处理文件夹中的所有.bmp图像并保存处理结果"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(params_dir):
        os.makedirs(params_dir)
    if not os.path.exists(line_dir):
        os.makedirs(line_dir)
    if not os.path.exists(json_dir):
        os.makedirs(json_dir)

    # 加载标定参数
    mtx, dist = load_calibration_params(calibration_folder)

    # 遍历输入文件夹中的所有.bmp文件
    for filename in os.listdir(input_dir):
        if filename.endswith('.bmp'):
            image_path = os.path.join(input_dir, filename)
            print(f"正在处理文件: {filename}")

            # 提取光条中心
            try:
                centers, output_image, adjusted_img, binary_img, roi_img = extract_light_stripe_gray_center_with_exposure(
                    image_path, alpha, beta, threshold, kernel_size, mtx, dist)

                # 保存光条中心坐标到CSV文件
                csv_path = os.path.join(params_dir, f"{filename.replace('.bmp', '_coordinates.csv')}")
                save_coordinates_to_csv(centers, csv_path)

                # 保存处理后的图像
                result_image_path = os.path.join(output_dir, f"processed_{filename}")
                cv2.imwrite(result_image_path, output_image)

                # 拟合直线并可视化，保存拟合直线参数
                save_path = os.path.join(output_dir, f"fit_{filename}.png")
                params_path = os.path.join(params_dir, f"{filename.replace('.bmp', '_params.csv')}")
                line_only_path = os.path.join(line_dir, f"line_{filename}")
                json_path = os.path.join(json_dir, f"{filename.replace('.bmp', '_line.json')}")  # JSON 文件路径
                fit_lines_and_visualize(centers, output_image, save_path, params_path, line_only_path, json_path, filename)

            except Exception as e:
                print(f"处理文件 {filename} 时出错: {e}")


# 输入和输出文件夹路径
input_folder = 'E:/calibrationtest/light_line'  # 输入文件夹路径，替换为实际路径
output_folder = 'E:/calibrationtest/output_pic'  # 输出文件夹路径，替换为实际路径
params_folder = 'E:/calibrationtest/params'  # 保存拟合参数和坐标的文件夹路径
line_folder = 'E:/calibrationtest/lines'  # 保存拟合直线图像的文件夹路径
json_folder = 'E:/calibrationtest/json_files'  # 保存拟合参数 JSON 的文件夹路径
calibration_folder = 'E:/calibrationtest/calibration_data'  # 相机标定参数文件路径

# 处理文件夹中的所有图片
process_directory(input_folder, output_folder, params_folder, line_folder, json_folder, alpha=8, beta=100, threshold=200,
                  kernel_size=5, calibration_folder=calibration_folder)
