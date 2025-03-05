import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

# 加载灰度图片
image_path = 'E:/calibrationtest/light_line/1846L-0.bmp'  # 替换为你的图片路径
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 定义竖直方向分块数量
num_vertical_blocks = 160  # 将图像竖直方向分成100个区块
h, w = image.shape
vertical_block_width = w // num_vertical_blocks  # 每个竖直区块的宽度

# 创建一个空白图像用于存储竖直过滤后的图像
filtered_image = np.zeros_like(image)

# 步骤1: 遍历竖直区块并计算黑色像素占比
for i in range(num_vertical_blocks):
    x_start, x_end = i * vertical_block_width, (i + 1) * vertical_block_width
    block = image[:, x_start:x_end]  # 整个高度的竖直区块

    # 计算该竖直区块的黑色像素占比
    black_pixel_count = np.sum(block < 5)  # 像素值小于20的视为黑色
    total_pixel_count = block.size
    black_ratio = black_pixel_count / total_pixel_count

    # 如果黑色像素占比小于999%，则保留该区块
    if black_ratio < 0.999999:
        filtered_image[:, x_start:x_end] = block

# 步骤2: 水平方向再划分
num_horizontal_blocks = 12  # 将保留的图像进一步划分成9个水平区块
horizontal_block_height = h // num_horizontal_blocks  # 每个水平区块的高度

# 创建一个空白图像来存储最终的二值化结果
binary_image = np.zeros_like(image, dtype=np.float32)  # 使用float32用于混合叠加

# 遍历每个水平区块和竖直区块，并进行多阈值处理
for i in range(num_vertical_blocks):
    x_start, x_end = i * vertical_block_width, (i + 1) * vertical_block_width
    for j in range(num_horizontal_blocks):
        y_start, y_end = j * horizontal_block_height, (j + 1) * horizontal_block_height
        block = filtered_image[y_start:y_end, x_start:x_end]  # 获取竖直区块内的水平区块

        # 水平区块内再次检测黑色像素占比，剔除全黑部分
        black_pixel_count = np.sum(block < 20)  # 像素值小于20的视为黑色
        total_pixel_count = block.size
        black_ratio = black_pixel_count / total_pixel_count

        # 如果水平区块内黑色像素占比超过99%，跳过该区块
        if black_ratio >= 0.99:
            continue  # 跳过全黑区域

        # 使用 Otsu 算法获得主阈值 (解包函数返回值)
        otsu_threshold, _ = cv2.threshold(block, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 计算该区块的亮度标准差，决定步进范围
        stddev = np.std(block)
        if stddev > 50:  # 如果标准差较大，说明区域对比度高，使用较大的步进
            step_range = 10
        else:  # 如果区域对比度低，使用较小的步进
            step_range = 5

        # 动态生成阈值列表，在 Otsu 阈值附近上下偏移
        thresholds = list(range(int(otsu_threshold) - step_range*2, int(otsu_threshold) + step_range*3, step_range))

        # 设置每个阈值对应的权重（根据需要可以调整权重策略）
        num_thresholds = len(thresholds)
        weights = np.linspace(0.2, 0.6, num_thresholds)  # 动态生成权重，从低到高

        # 累加权重后的二值化结果
        for t, weight in zip(thresholds, weights):
            # 确保阈值 t 是浮点数，并进行二值化操作
            _, block_binary = cv2.threshold(block, float(t), 255, cv2.THRESH_BINARY)
            binary_image[y_start:y_end, x_start:x_end] += weight * block_binary / 255.0  # 归一化后叠加

# 步骤3: 将叠加结果二值化，确保输出为纯黑白图像
binary_image[binary_image >= 0.5] = 1  # 将灰度值大于等于 0.5 的区域设为 1（白）
binary_image[binary_image < 0.5] = 0   # 将灰度值小于 0.5 的区域设为 0（黑）

# 步骤4: 将结果转换回二值化图像格式
final_binary_image = (binary_image * 255).astype(np.uint8)  # 乘回255并转换为uint8

# 显示原始图片和处理后的混合二值化图片
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(image, cmap='gray')
ax[0].set_title("Original Image")
ax[0].axis('off')

ax[1].imshow(final_binary_image, cmap='gray')
ax[1].set_title("Binarized Image with HDR Effect")
ax[1].axis('off')

plt.show()

# 保存结果图片
binary_image_path = 'E:/guangtiaotiqu/pic/hdr_binarized_image.jpg'  # 替换为你的保存路径
cv2.imwrite(binary_image_path, final_binary_image)