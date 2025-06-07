import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ====================================== 读取图像 ======================================
def read_image(path):
    img = Image.open(path).convert('L')  # 转为灰度图
    return np.array(img, dtype=np.float32) / 255.0  # 归一化到[0,1]

# ====================================== 手动DFT ======================================
def dft_matrix(N):
    x, u = np.meshgrid(np.arange(N), np.arange(N))
    omega = np.exp(-2j * np.pi * x * u / N)
    return omega / np.sqrt(N)  # 标准化

def dft2d(image):
    N, M = image.shape
    # 构建行和列的DFT矩阵
    W_row = dft_matrix(N)
    W_col = dft_matrix(M)
    # 行变换 → 列变换
    dft_rows = np.dot(W_row, image)
    dft_result = np.dot(dft_rows, W_col)
    return dft_result

# ====================================== 手动DCT ======================================
def dct_matrix(N):
    c = np.zeros((N, N))
    for u in range(N):
        for x in range(N):
            if u == 0:
                c[u, x] = np.sqrt(1/N)
            else:
                c[u, x] = np.sqrt(2/N) * np.cos((2*x + 1) * u * np.pi / (2*N))
    return c

def dct2d(image):
    N, M = image.shape
    # 构建DCT矩阵
    C_row = dct_matrix(N)
    C_col = dct_matrix(M)
    # 行变换 → 列变换
    dct_rows = np.dot(C_row, image)
    dct_result = np.dot(dct_rows, C_col.T)
    return dct_result

# ====================================== 频谱可视化 ======================================
def plot_spectrum(dft, dct):
    plt.figure(figsize=(12, 4))
    
    # DFT频谱（对数幅度 + 中心化）
    dft_shift = np.fft.fftshift(dft)
    magnitude = np.log(np.abs(dft_shift) + 1e-9)  # 避免log(0)
    plt.subplot(131), plt.imshow(magnitude, cmap='gray'), plt.title('DFT Spectrum')
    
    # DCT频谱（直接显示）
    plt.subplot(132), plt.imshow(np.abs(dct), cmap='gray'), plt.title('DCT Spectrum')
    
    plt.tight_layout()
    plt.show()

# ====================================== 主程序 ======================================
if __name__ == "__main__":
    # 读取图像（需替换为你的lena.bmp路径）
    image = read_image("lena.bmp")
    
    # 计算DFT和DCT
    dft = dft2d(image)
    dct = dct2d(image)
    
    # 显示原图和频谱
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap='gray'), plt.title('Original Image')
    plot_spectrum(dft, dct)