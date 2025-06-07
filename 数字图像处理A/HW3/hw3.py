import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread('Lena.bmp', cv2.IMREAD_GRAYSCALE)

# --------------------- 添加噪声 ---------------------
def add_gaussian_noise(img, mean=0, sigma=25):
    """添加高斯噪声[4,5](@ref)"""
    noise = np.random.normal(mean, sigma, img.shape).astype(np.int16)
    noisy_img = np.clip(img + noise, 0, 255).astype(np.uint8)
    return noisy_img

def add_salt_pepper_noise(img, prob=0.05):
    """快速椒盐噪声[2,3](@ref)"""
    mask = np.random.choice([0, 255], size=img.shape, p=[1-prob, prob])
    salt_mask = (mask == 255)
    pepper_mask = (mask == 0)
    noisy_img = img.copy()
    noisy_img[salt_mask] = 255
    noisy_img[pepper_mask] = 0
    return noisy_img

# 生成噪声图像
gaussian_img = add_gaussian_noise(image)
sp_img = add_salt_pepper_noise(image)

# --------------------- 滤波处理 ---------------------
# 局部平均滤波（均值滤波）[7,8](@ref)
mean_filtered_gauss = cv2.blur(gaussian_img, (5, 5))
mean_filtered_sp = cv2.blur(sp_img, (5, 5))

# 中值滤波[6,13](@ref)
median_filtered_gauss = cv2.medianBlur(gaussian_img, 5)
median_filtered_sp = cv2.medianBlur(sp_img, 5)

# --------------------- 可视化 ---------------------
plt.figure(figsize=(12, 8))

# 原始图像与噪声图像
plt.subplot(2, 3, 1), plt.imshow(image, cmap='gray'), plt.title('Original')
plt.subplot(2, 3, 2), plt.imshow(gaussian_img, cmap='gray'), plt.title('Gaussian Noise')
plt.subplot(2, 3, 3), plt.imshow(sp_img, cmap='gray'), plt.title('Salt & Pepper Noise')

# 滤波结果
plt.subplot(2, 3, 4), plt.imshow(mean_filtered_gauss, cmap='gray'), plt.title('Mean Filter (Gaussian)')
plt.subplot(2, 3, 5), plt.imshow(median_filtered_gauss, cmap='gray'), plt.title('Median Filter (Gaussian)')
plt.subplot(2, 3, 6), plt.imshow(median_filtered_sp, cmap='gray'), plt.title('Median Filter (Salt & Pepper)')

plt.tight_layout()
plt.show()