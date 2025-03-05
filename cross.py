import cv2
import os
import json

def load_line_params(json_path, is_light_line=False):
    """
    从 JSON 文件加载直线拟合参数。
    :param json_path: JSON 文件路径
    :param is_light_line: 如果是光条直线，则特殊处理
    :return: 返回线的参数列表
    """
    with open(json_path, 'r') as f:
        lines = json.load(f)

    if is_light_line:
        # 解析光条直线格式
        slope = lines["line_params"]["slope"]
        intercept = lines["line_params"]["intercept"]
        # 将斜率和截距转换为方向向量形式
        vx = 1
        vy = slope
        x0 = 0
        y0 = intercept
        return [{"vx": vx, "vy": vy, "x0": x0, "y0": y0}]
    else:
        # 默认处理棋盘格线
        return lines


def calculate_intersection(line1, line2):
    """
    计算两条直线的交点。
    :param line1: 第一条直线参数 (vx, vy, x0, y0)
    :param line2: 第二条直线参数 (vx, vy, x0, y0)
    :return: 交点坐标 (x, y)，如果平行则返回 None
    """
    vx1, vy1, x01, y01 = line1["vx"], line1["vy"], line1["x0"], line1["y0"]
    vx2, vy2, x02, y02 = line2["vx"], line2["vy"], line2["x0"], line2["y0"]

    # 计算交点
    denominator = vx1 * vy2 - vy1 * vx2
    if denominator == 0:
        return None  # 平行或重合

    t1 = ((x02 - x01) * vy2 - (y02 - y01) * vx2) / denominator
    x = x01 + t1 * vx1
    y = y01 + t1 * vy1
    return x, y


def draw_intersections(image_path, chessboard_lines, light_lines, save_path, points_save_path):
    """
    计算光条与棋盘格线的交点，并在棋盘格图像上绘制，同时保存交点为 JSON 文件。
    :param image_path: 棋盘格图像路径
    :param chessboard_lines: 棋盘格线参数列表
    :param light_lines: 光条直线参数列表
    :param save_path: 保存结果图像路径
    :param points_save_path: 保存交点 JSON 文件路径
    """
    # 加载棋盘格图像
    print(f"Loading image: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法加载图像: {image_path}")
    print(f"Image loaded successfully with shape: {img.shape}")

    # 检查图像是否是灰度图
    if len(img.shape) == 2:  # 如果是单通道（灰度图像）
        print("Image is grayscale, converting to BGR.")
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        print("Image is already BGR.")
        img_color = img  # 已经是彩色图像

    # 用于保存交点的列表
    intersections = []

    # 绘制光条直线
    for light_line in light_lines:
        vx, vy, x0, y0 = light_line["vx"], light_line["vy"], light_line["x0"], light_line["y0"]
        slope = vy / vx
        intercept = y0 - slope * x0

        # 计算光条直线的两个端点（与图像边界的交点）
        x1, y1 = 0, int(intercept)
        x2, y2 = img_color.shape[1] - 1, int(slope * (img_color.shape[1] - 1) + intercept)

        # 绘制光条直线（蓝色）
        cv2.line(img_color, (x1, y1), (x2, y2), (255, 0, 0), 1)  # 蓝色线条

    # 绘制棋盘格线
    for chess_line in chessboard_lines:
        vx, vy, x0, y0 = chess_line["vx"], chess_line["vy"], chess_line["x0"], chess_line["y0"]
        slope = vy / vx
        intercept = y0 - slope * x0

        # 计算棋盘格直线的两个端点（与图像边界的交点）
        x1, y1 = 0, int(intercept)
        x2, y2 = img_color.shape[1] - 1, int(slope * (img_color.shape[1] - 1) + intercept)

        # 绘制棋盘格直线（绿色）
        cv2.line(img_color, (x1, y1), (x2, y2), (0, 255, 0), 1)  # 绿色线条

    # 遍历每一条光条和棋盘格线，计算交点
    for light_line in light_lines:
        for chess_line in chessboard_lines:
            intersection = calculate_intersection(light_line, chess_line)
            if intersection:
                x, y = intersection
                print(f"Intersection found at: ({x}, {y})")  # 调试输出
                intersections.append({"x": x, "y": y})  # 保存交点坐标
                # 在图像上绘制交点（红色）
                cv2.circle(img_color, (int(x), int(y)), 2, (0, 0, 255), -1)  # 红色交点

    # 保存结果图像
    cv2.imwrite(save_path, img_color)
    print(f"结果图像已保存到: {save_path}")

    # 保存交点为 JSON 文件
    with open(points_save_path, 'w') as f:
        json.dump(intersections, f, indent=4)
    print(f"交点坐标已保存到: {points_save_path}")


def process_directory(image_dir, results_dir, json_dir, output_dir, points_dir):
    """
    遍历图像文件夹中的所有 .bmp 文件，并处理对应的 JSON 文件。
    :param image_dir: 图像文件夹路径
    :param results_dir: 棋盘格线 JSON 文件夹路径
    :param json_dir: 光条直线 JSON 文件夹路径
    :param output_dir: 保存结果图像的文件夹
    :param points_dir: 保存交点 JSON 文件的文件夹
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(points_dir):
        os.makedirs(points_dir)

    # 遍历图像文件夹中的所有 .bmp 文件
    for filename in os.listdir(image_dir):
        if filename.endswith(".bmp"):
            # 获取图像路径
            image_path = os.path.join(image_dir, filename)

            # 根据文件名推导 JSON 文件路径
            base_name = os.path.splitext(filename)[0]
            chessboard_lines_json = os.path.join(results_dir, f"{base_name}_lines.json")

            # 为光条文件插入 "L" 前缀
            light_lines_base_name = f"{base_name[:4]}L{base_name[4:]}"
            light_lines_json = os.path.join(json_dir, f"{light_lines_base_name}_line.json")
            save_path = os.path.join(output_dir, f"{base_name}_output.png")
            points_save_path = os.path.join(points_dir, f"{base_name}_intersections.json")

            # 检查 JSON 文件是否存在
            if not os.path.exists(chessboard_lines_json):
                print(f"棋盘格线 JSON 文件不存在: {chessboard_lines_json}")
                continue
            if not os.path.exists(light_lines_json):
                print(f"光条直线 JSON 文件不存在: {light_lines_json}")
                continue

            # 加载直线参数
            try:
                chessboard_lines = load_line_params(chessboard_lines_json)
                light_lines = load_line_params(light_lines_json, is_light_line=True)

                # 计算交点并保存结果
                draw_intersections(image_path, chessboard_lines, light_lines, save_path, points_save_path)
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {e}")


def main():
    # 输入路径
    image_dir = "E:/calibrationtest/pic"  # 图像文件夹路径
    results_dir = "E:/calibrationtest/results"  # 棋盘格线 JSON 文件夹路径
    json_dir = "E:/calibrationtest/json_files"  # 光条直线 JSON 文件夹路径
    output_dir = "E:/calibrationtest/outputs"  # 输出文件夹路径
    points_dir = "E:/calibrationtest/intersections"  # 保存交点 JSON 文件的文件夹

    # 遍历并处理目录中的所有 .bmp 文件
    process_directory(image_dir, results_dir, json_dir, output_dir, points_dir)



if __name__ == "__main__":
    main()
