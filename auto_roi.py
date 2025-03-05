import cv2
import numpy as np
import os


def select_roi_once(image_path):
    """人工选择ROI，并保存选择的区域为模板。"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("无法加载图像，请检查路径！")

    # 手动选择ROI
    roi = cv2.selectROI("选择光条区域", img)
    cv2.destroyAllWindows()

    # 获取ROI的区域
    x, y, w, h = map(int, roi)
    roi_img = img[y:y + h, x:x + w]

    # 保存ROI模板
    return roi_img, x, y, w, h


def template_matching(image_path, template_img):
    """使用模板匹配方法在图像中查找ROI。"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("无法加载图像，请检查路径！")

    # 使用cv2.matchTemplate进行模板匹配
    result = cv2.matchTemplate(img, template_img, cv2.TM_CCOEFF_NORMED)

    # 找到最佳匹配的位置
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # 获得匹配的区域
    x, y = max_loc
    h, w = template_img.shape
    return (x, y, w, h)


def process_directory_with_template(input_dir, output_dir, template_img):
    """遍历文件夹，使用模板匹配在图像中自动选择ROI。"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith('.bmp'):
            image_path = os.path.join(input_dir, filename)
            print(f"正在处理文件: {filename}")

            try:
                # 使用模板匹配进行自动ROI选择
                x, y, w, h = template_matching(image_path, template_img)

                # 读取原图并显示结果
                img = cv2.imread(image_path)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # 保存带框选结果的图像
                result_image_path = os.path.join(output_dir, f"processed_{filename}")
                cv2.imwrite(result_image_path, img)

            except Exception as e:
                print(f"处理文件 {filename} 时出错: {e}")


# 选择第一次ROI并保存模板
template_img, x, y, w, h = select_roi_once('E:/calibrationtest/light_line/1846L-00.bmp')

# 输入和输出文件夹路径
input_folder = 'E:/calibrationtest/light_line'  # 输入文件夹路径，替换为实际路径
output_folder = 'E:/calibrationtest/output'  # 输出文件夹路径，替换为实际路径

# 使用模板匹配在文件夹中的所有图像中选择ROI并处理
process_directory_with_template(input_folder, output_folder, template_img)
