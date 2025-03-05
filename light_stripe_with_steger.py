import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

font_path = 'C:/Windows/Fonts/simhei.ttf'
font_prop = font_manager.FontProperties(fname=font_path)

def adjust_exposure(img, alpha=5, beta=100):
    """调整图像曝光。"""
    adjusted_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return adjusted_img

def steger_line_detection(img, sigma=1.0):
    """使用 Steger 算法提取光条中心线。"""
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    gx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=7)
    gy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=7)
    gxx = cv2.Sobel(gx, cv2.CV_64F, 1, 0, ksize=5)
    gxy = cv2.Sobel(gx, cv2.CV_64F, 0, 1, ksize=5)
    gyy = cv2.Sobel(gy, cv2.CV_64F, 0, 1, ksize=5)

    det = gxx * gyy - gxy * gxy
    trace = gxx + gyy
    lambda1 = trace / 2 + np.sqrt((trace / 2) ** 2 - det)
    lambda2 = trace / 2 - np.sqrt((trace / 2) ** 2 - det)

    centers = []
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if np.abs(lambda2[y, x]) > np.abs(lambda1[y, x]):
                dx = -gy[y, x]
                dy = gx[y, x]
                norm = np.sqrt(dx ** 2 + dy ** 2)
                if norm > 0:
                    dx /= norm
                    dy /= norm
                centers.append((x + dx, y + dy))
    return centers

def select_roi(img):
    """手动框选感兴趣区域 (ROI)。"""
    roi = cv2.selectROI("选择光条区域", img)
    cv2.destroyAllWindows()
    if roi[2] == 0 or roi[3] == 0:
        raise ValueError("未选择有效的ROI区域！")
    x, y, w, h = map(int, roi)
    return img[y:y+h, x:x+w], x, y, w, h

def extract_light_stripe_with_steger(image_path, alpha=8, beta=100, sigma=10):
    """使用 Steger 算法提取光条中心线。"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("无法加载图像，请检查路径！")

    roi_img, x_offset, y_offset, _, _ = select_roi(img)
    adjusted_img = adjust_exposure(roi_img, alpha=alpha, beta=beta)
    centers = steger_line_detection(adjusted_img, sigma=sigma)

    output_image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for center_x, center_y in centers:
        original_x = int(center_x) + x_offset
        original_y = int(center_y) + y_offset
        cv2.circle(output_image, (original_x, original_y), 1, (0, 0, 255), -1)

    return centers, output_image, adjusted_img, roi_img

# 加载图像并运行
image_path = 'E:/calibrationtest/light_line/1846L-00.bmp'
alpha, beta, sigma = 8, 100, 10

centers, output_image, adjusted_img, roi_img = extract_light_stripe_with_steger(
    image_path, alpha=alpha, beta=beta, sigma=sigma)

# 保存并可视化结果
output_result_path = "light_stripe_with_steger_coordinates_no_binarization.bmp"
cv2.imwrite(output_result_path, output_image)

plt.figure(figsize=(20, 15))
plt.subplot(1, 3, 1)
plt.imshow(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE), cmap='gray')
plt.title("原始图像", fontproperties=font_prop)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(adjusted_img, cmap='gray')
plt.title("曝光调整后的图像", fontproperties=font_prop)
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
plt.title("提取的光条中心线 (Steger)", fontproperties=font_prop)
plt.axis("off")

plt.tight_layout()
plt.show()

print(f"光条中心点坐标（共{len(centers)}个点）：")
for point in centers:
    print(point)
