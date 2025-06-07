import numpy as np
from PIL import Image
import math

def rotate_image_grayscale(img_path, angle_deg, output_path):
    """
    任意角度旋转灰度图像
    参数：
        img_path: 输入图像路径
        angle_deg: 旋转角度（度数）
        output_path: 输出图像路径
    """
    # 读取原始图像
    img = Image.open(img_path).convert('L')  # 转换为灰度图
    img_array = np.array(img)
    h, w = img_array.shape

    # 将角度转换为弧度
    angle_rad = math.radians(angle_deg)
    cos_theta = math.cos(angle_rad)
    sin_theta = math.sin(angle_rad)

    # 计算旋转后的图像尺寸
    # 获取四个角点旋转后的坐标
    corners = np.array([
        [0, 0],
        [w-1, 0],
        [0, h-1],
        [w-1, h-1]
    ])

    # 计算旋转后的坐标
    cx, cy = w/2, h/2  # 原图中心
    rotated_corners = []
    for x, y in corners:
        # 平移坐标系到中心点
        x_rel = x - cx
        y_rel = y - cy
        # 应用旋转矩阵
        x_rot = x_rel * cos_theta - y_rel * sin_theta
        y_rot = x_rel * sin_theta + y_rel * cos_theta
        # 移回原坐标系
        rotated_corners.append([x_rot + cx, y_rot + cy])

    # 计算新图像尺寸
    rotated_corners = np.array(rotated_corners)
    x_min, y_min = np.min(rotated_corners, axis=0)
    x_max, y_max = np.max(rotated_corners, axis=0)
    new_w = int(math.ceil(x_max - x_min))
    new_h = int(math.ceil(y_max - y_min))

    # 创建新图像数组（初始化全黑）
    rotated_img = np.zeros((new_h, new_w), dtype=np.uint8)
    new_cx, new_cy = new_w/2, new_h/2  # 新图中心

    # 遍历新图像的每个像素
    for y_new in range(new_h):
        for x_new in range(new_w):
            # 将新坐标转换到原图坐标系
            # 平移坐标系到新中心
            x_rel = x_new - new_cx
            y_rel = y_new - new_cy
            # 应用逆向旋转矩阵
            x_ori = x_rel * cos_theta + y_rel * sin_theta
            y_ori = -x_rel * sin_theta + y_rel * cos_theta
            # 移回原图坐标系
            x_ori += cx
            y_ori += cy

            # 检查是否在原图范围内
            if 0 <= x_ori < w and 0 <= y_ori < h:
                # 双线性插值
                x0 = int(math.floor(x_ori))
                y0 = int(math.floor(y_ori))
                x1 = min(x0 + 1, w - 1)
                y1 = min(y0 + 1, h - 1)

                # 计算权重
                dx = x_ori - x0
                dy = y_ori - y0

                # 获取四个相邻像素值
                val = (1 - dx) * (1 - dy) * img_array[y0, x0] + \
                      dx * (1 - dy) * img_array[y0, x1] + \
                      (1 - dx) * dy * img_array[y1, x0] + \
                      dx * dy * img_array[y1, x1]

                rotated_img[y_new, x_new] = int(val)

    # 保存结果
    Image.fromarray(rotated_img).save(output_path)

# 使用示例
if __name__ == "__main__":
    rotate_image_grayscale(
        img_path = "lena.bmp",
        angle_deg = 45,       # 旋转45度
        output_path = "rotated_lema.bmp"
    )