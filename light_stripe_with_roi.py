import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

font_path = 'C:/Windows/Fonts/simhei.ttf'
font_prop = font_manager.FontProperties(fname=font_path)

def adjust_exposure_and_threshold(img, alpha=1.5, beta=50, threshold=200):
    """
    调整图像曝光并进行二值化处理。

    参数:
    - img: 输入图像。
    - alpha: 增益系数，用于控制对比度。
    - beta: 加亮值，用于增加亮度。
    - threshold: 二值化阈值，用于提取亮区域。

    返回:
    - adjusted_img: 曝光调整后的图像。
    - binary_img: 二值化后的图像。
    """
    adjusted_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    _, binary_img = cv2.threshold(adjusted_img, threshold, 255, cv2.THRESH_BINARY)
    return adjusted_img, binary_img

def select_roi(img):
    """
    手动框选感兴趣区域 (ROI)。

    参数:
    - img: 输入图像。

    返回:
    - roi: 框选区域的图像。
    - x, y, w, h: 框选区域的原始坐标。
    """
    print("请手动框选光条所在区域，按回车确认，按 ESC 取消")
    roi = cv2.selectROI("选择光条区域", img)
    cv2.destroyAllWindows()
    x, y, w, h = map(int, roi)
    return img[y:y+h, x:x+w], x, y, w, h

def extract_light_stripe_gray_center_with_exposure(image_path, alpha=1.5, beta=50, threshold=200, kernel_size=5):
    """
    使用增加曝光预处理和灰度中心法提取光条中心线。

    参数:
    - image_path: 输入图像路径。
    - alpha: 曝光调整的增益系数。
    - beta: 曝光调整的加亮值。
    - threshold: 二值化阈值，用于提取亮区域。
    - kernel_size: 平滑处理的核大小。

    返回:
    - centers: 光条中心点的原始坐标列表。
    - output_image: 标记光条中心线后的原始图像。
    - adjusted_img: 曝光调整后的图像。
    - binary_img: 二值化后的图像。
    """
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

# 加载灰度图片
image_path = 'E:/calibrationtest/light_line/1846L-0.bmp'  # 替换为你的图片路径
alpha = 8  # 增益系数
beta = 100  # 加亮值
threshold_value = 200  # 二值化阈值
kernel_size = 5  # 平滑处理的核大小

centers, output_image, adjusted_img, binary_img, roi_img = extract_light_stripe_gray_center_with_exposure(
    image_path, alpha=alpha, beta=beta, threshold=threshold_value, kernel_size=kernel_size)

# 保存和显示结果
output_result_path = "light_stripe_with_original_coordinates.bmp"
cv2.imwrite(output_result_path, output_image)

# 可视化结果
plt.figure(figsize=(20, 15))

# 原始图像
plt.subplot(2, 3, 1)
plt.imshow(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE), cmap='gray')
plt.title("原始图像", fontproperties=font_prop)
plt.axis("off")

# 手动选择的区域
plt.subplot(2, 3, 2)
plt.imshow(roi_img, cmap='gray')
plt.title("手动选择的区域 (ROI)", fontproperties=font_prop)
plt.axis("off")

# 曝光调整后的图像
plt.subplot(2, 3, 3)
plt.imshow(adjusted_img, cmap='gray')
plt.title("曝光调整后的图像", fontproperties=font_prop)
plt.axis("off")

# 二值化图像
plt.subplot(2, 3, 4)
plt.imshow(binary_img, cmap='gray')
plt.title("二值化后的图像", fontproperties=font_prop)
plt.axis("off")

# 提取的光条中心线
plt.subplot(2, 3, 5)
plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
plt.title("提取的光条中心线 (原始坐标)", fontproperties=font_prop)
plt.axis("off")

plt.tight_layout()
plt.show()

print(f"光条中心点坐标（共{len(centers)}个点）：")
for point in centers:
    print(point)
