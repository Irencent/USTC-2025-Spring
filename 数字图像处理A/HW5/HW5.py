import numpy as np
import matplotlib.pyplot as plt
from pylab import *
mpl.rcParams['font.sans-serif'] = ['Heiti TC']  # 设置中文字体

# 定义运动模糊参数
L = 9  # 水平运动模糊的像素长度
psf_length = L + 1  # PSF的离散长度
psf = np.zeros(psf_length)
psf[:] = 1.0 / psf_length  # 归一化权重

# 绘制PSF曲线
positions = np.arange(psf_length)  # 像素位置

plt.figure(figsize=(8, 4))
plt.stem(positions, psf, linefmt='b-', markerfmt='bo', basefmt=' ')
plt.title(f'水平匀速运动模糊PSF (L={L}像素)')
plt.xlabel('像素位置')
plt.ylabel('幅值')
plt.xticks(positions)
plt.ylim(0, 0.12)  # 根据L=9时幅值为0.1调整
plt.grid(True)
plt.show()