from PIL import Image

def process_image(image_path):
    try:
        # 打开图像并强制转换为灰度模式
        img = Image.open(image_path).convert('L')
    except FileNotFoundError:
        print(f"错误：文件 '{image_path}' 未找到！")
        return
    except Exception as e:
        print(f"打开图像失败：{e}")
        return

    width, height = img.size
    print(f"原始图像尺寸：{width}x{height} (宽x高)")

    # 检查图像高度是否足够
    if height < 256:
        print(f"错误：图像高度不足 256 行（实际高度：{height}）")
        return

    # 获取像素操作接口
    pixels = img.load()

    # 将前 256 行设为纯白（灰度值 255）
    for y in range(256):
        for x in range(width):
            pixels[x, y] = 255  # 修改坐标为 (x,y) 的像素

    # 显示处理后的图像
    img.show()
    print("已显示处理后的图像（前256行为白色）")

def pringPixel(image_path):
    try:
        img = Image.open(image_path)
        if img.mode != 'L':
            img = img.convert('L')  # 转换为灰度图像
    except FileNotFoundError:
        print(f"错误：文件 '{image_path}' 未找到！")
        exit()
    except Exception as e:
        print(f"打开图像失败：{e}")
        exit()

    # 检查图像尺寸是否合法
    width, height = img.size
    x_start, y_start = 200, 200
    region_size = 10

    if (x_start + region_size > width) or (y_start + region_size > height):
        print(f"错误：区域超出图像范围（图像尺寸：{width}x{height}）")
        exit()

    # 截取 10x10 区域像素值（高效方法）
    region = img.crop((x_start, y_start, x_start + region_size, y_start + region_size))
    pixel_data = list(region.getdata())  # 将像素值转换为列表

    # 将一维列表转换为二维矩阵（10x10）
    pixel_matrix = [pixel_data[i*region_size : (i+1)*region_size] for i in range(region_size)]

    # 打印结果
    print(f"以 ({x_start}, {y_start}) 为左上角的 10x10 区域像素值：")
    for row in pixel_matrix:
        print(' '.join(f"{val:3}" for val in row))  # 格式化输出

import numpy as np

def apply_colormap(gray_image, colormap_func):
    """
    将灰度图像通过自定义颜色映射函数转换为伪彩色图像
    :param gray_image: 灰度图像（PIL Image，模式为'L'）
    :param colormap_func: 颜色映射函数，输入灰度值（0-255），返回RGB元组
    :return: 伪彩色图像（PIL Image，模式为'RGB'）
    """
    # 将灰度图像转为NumPy数组
    gray_array = np.array(gray_image, dtype=np.uint8)
    
    # 应用颜色映射函数（向量化操作）
    rgb_array = np.zeros((*gray_array.shape, 3), dtype=np.uint8)
    for i in range(gray_array.shape[0]):
        for j in range(gray_array.shape[1]):
            gray_value = gray_array[i, j]
            rgb_array[i, j] = colormap_func(gray_value)
    
    # 转为PIL图像
    return Image.fromarray(rgb_array, 'RGB')

def rainbow_colormap(gray_value):
    """
    彩虹渐变颜色映射函数（示例）
    :param gray_value: 灰度值（0-255）
    :return: (R, G, B)元组（0-255）
    """
    # 将灰度值映射到HSV的Hue分量（0-0.7范围，红→黄→绿→青→蓝→紫）
    hue = gray_value / 255 * 0.7  # 限制Hue范围避免颜色循环重复
    saturation = 1.0  # 最大饱和度
    value = 1.0        # 最大亮度
    
    # HSV转RGB
    from colorsys import hsv_to_rgb
    r, g, b = hsv_to_rgb(hue, saturation, value)
    return (int(r * 255), int(g * 255), int(b * 255))

# 主程序
if __name__ == "__main__":
    image_path = "lena.bmp"
    pringPixel(image_path)

    # try:
    #     # 打开图像并强制转换为灰度模式
    #     img = Image.open("lena.bmp").convert('L')
    #     print(f"图像尺寸：{img.size}，模式：{img.mode}")
        
    #     # 应用彩虹颜色映射
    #     color_img = apply_colormap(img, rainbow_colormap)
        
    #     # 显示处理后的图像
    #     color_img.show()
    #     print("已显示伪彩色图像（彩虹渐变）")
        
    #     # 可选：保存结果
    #     color_img.save("colored_lena.png")
    # except Exception as e:
    #     print(f"错误：{e}")

