import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 设置支持中文的字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
font_path = 'C:/Windows/Fonts/simhei.ttf'

font_prop = font_manager.FontProperties(fname=font_path)

def adjust_exposure_and_threshold(img, alpha=1.5, beta=50, threshold=200):
    """调整图像曝光并进行二值化处理。"""
    adjusted_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    _, binary_img = cv2.threshold(adjusted_img, threshold, 255, cv2.THRESH_BINARY)
    return adjusted_img, binary_img

def select_roi(img):
    """手动框选感兴趣区域 (ROI)。"""
    print("请手动框选光条所在区域，按回车确认，按 ESC 取消")
    roi = cv2.selectROI("选择光条区域", img)
    cv2.destroyAllWindows()
    x, y, w, h = map(int, roi)
    return img[y:y+h, x:x+w], x, y, w, h

def extract_light_stripe_gray_center_with_exposure(image_path, alpha=1.5, beta=50, threshold=200, kernel_size=5):
    """使用增加曝光预处理和灰度中心法提取光条中心线。"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("无法加载图像，请检查路径！")

    # 手动框选感兴趣区域 (ROI)
    roi_img, x_offset, y_offset, _, _ = select_roi(img)

    # 曝光调整和二值化
    adjusted_img, binary_img = adjust_exposure_and_threshold(roi_img, alpha=alpha, beta=beta, threshold=threshold)

    # 平滑图像以减少噪声
    blurred = cv2.GaussianBlur(adjusted_img, (kernel_size, kernel_size), 0)

    # 提取光条中心点
    centers = []
    output_image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

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

def fit_lines_and_visualize(centers, output_image):
    """对提取的点进行直线拟合并可视化。"""
    # 将中心点分成 x 和 y 坐标
    points = np.array(centers)
    x_coords, y_coords = points[:, 0], points[:, 1]

    # 直线拟合 (y = mx + c)
    fit_params = np.polyfit(x_coords, y_coords, 1)  # 一阶多项式拟合
    m, c = fit_params

    # 计算拟合直线的两个端点
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = m * x_min + c, m * x_max + c

    # 可视化
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    plt.scatter(x_coords, y_coords, color='red', s=1, label='提取的中心点')
    plt.plot([x_min, x_max], [y_min, y_max], color='blue', label='拟合直线', linewidth=1)
    plt.title("光条中心点与拟合直线", fontproperties=font_prop)
    plt.legend()
    plt.axis("off")
    plt.show()

# 加载灰度图片
image_path = 'E:/calibrationtest/light_line/1846L-00.bmp'  # 替换为你的图片路径
alpha = 8  # 增益系数
beta = 100  # 加亮值
threshold_value = 200  # 二值化阈值
kernel_size = 5  # 平滑处理的核大小

centers, output_image, adjusted_img, binary_img, roi_img = extract_light_stripe_gray_center_with_exposure(
    image_path, alpha=alpha, beta=beta, threshold=threshold_value, kernel_size=kernel_size)

# 保存和显示结果
output_result_path = "light_stripe_with_fitted_line.bmp"
cv2.imwrite(output_result_path, output_image)

# 对提取的点拟合直线并可视化
fit_lines_and_visualize(centers, output_image)
