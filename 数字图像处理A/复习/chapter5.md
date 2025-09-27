**第5章 图像复原 - 笔记总结**

---

### 5.1 引言

**目的**：

* 通过补偿或还原图像的退化，恢复原始图像。

**退化原因**：

1. 大气扰动
2. 采样与量化
3. 运动模糊
4. 摄像头失焦
5. 噪声

**退化模型**：
$g(x, y) = f(x, y) * h(x, y) + n(x, y)$

* $f(x, y)$：原始图像
* $h(x, y)$：点扩散函数（PSF）
* $n(x, y)$：加性噪声

**复原方法**：

* 非约束方法：逆滤波（Inverse filtering）
* 有约束方法：维纳滤波（Wiener filtering）

---

### 5.2 对角化（Diagonalization）

**一维矩阵模型**：
$g = Hf + n$

* $H$：由PSF构成的循环矩阵

**二维矩阵模型**：
$g = Hf + n$

* $H$：块循环矩阵

**通过傅里叶变换进行对角化**：

* 一维：
  $H = W D W^{-1}$
  $G = D F + N$
* 二维：
  $G(u,v) = F(u,v) H(u,v) + N(u,v)$

---

### 5.3 逆滤波（Inverse Filtering）

**前提假设**：

* 已知H，且忽略噪声

**复原公式**：
$\hat{F}(u,v) = \frac{G(u,v)}{H(u,v)}$

**问题**：

* 当 $H(u,v)$ 接近零时对噪声非常敏感

**改进方法**：

* 使用阈值限制逆操作：
  $M(u,v) = \begin{cases} \frac{1}{H(u,v)} & \text{若 } |H(u,v)| > \delta \\ 0 & \text{否则} \end{cases}$

---

### 5.4 维纳滤波（Wiener Filtering）

**前提假设**：

* 已知H，噪声存在

**目标**：

* 最小化均方误差：
  $e^2 = E\{ (f - \hat{f})^2 \}$

**维纳滤波公式**：
$\hat{F}(u,v) = \left[ \frac{H^*(u,v)}{|H(u,v)|^2 + \frac{S_n(u,v)}{S_f(u,v)}} \right] G(u,v)$

* $S_n, S_f$：噪声与原图的功率谱

**简化形式**：
$H_w(u,v) = \frac{H^*(u,v)}{|H(u,v)|^2 + K}$

---

### 5.5 退化函数估计（Estimating the Degradation Function）

**1. 图像观测法**：
$H(u,v) = \frac{G(u,v)}{F(u,v)}$

**2. 实验测量法**：

* 脉冲响应：
  $f(x,y) = \delta(x,y) \Rightarrow g(x,y) = h(x,y)$

**3. 模型推导法**：

* 大气扰动模型：
  $H(u,v) = e^{-k(u^2 + v^2)^{5/6}}$
* 匀速直线运动模糊：
  $H(u,v) = \frac{\sin(\pi (ua + vb))}{\pi (ua + vb)} e^{-j\pi (ua + vb)}$

---

### 5.6 几何畸变校正（Geometric Distortion Correction）

**变换模型**：
$g(x,y) = f(r(x,y), s(x,y))$

* 示例：

  * 缩小：$x' = x/2, y' = y/2$
  * 放大：$x' = 2x, y' = 2y$
  * 旋转：
    $x' = x \cos\theta - y \sin\theta$
    $y' = x \sin\theta + y \cos\theta$

**空间变换类型**：

* 线性：
  $r = a_1x + a_2y + a_3, \quad s = b_1x + b_2y + b_3$
* 二次：
  $r = a_1x + a_2y + a_3xy + a_4x^2 + a_5y^2 + a_6$

**灰度插值方法**：

* 最近邻插值：
  $f(x', y') = f(\text{round}(x'), \text{round}(y'))$
* 双线性插值：
  $f(x', y') = (1-u)(1-v)f(i,j) + (1-u)v f(i,j+1) + u(1-v)f(i+1,j) + uvf(i+1,j+1)$

---

### 5.7 图像修复（Image Inpainting）

* 目的：对图像中的缺失或损坏区域进行填补。
* 方法：基于扩散、插值或偏微分方程模型。

---

### 总结

* 逆滤波：简单但对噪声敏感
* 维纳滤波：最小均方误差的最优方案
* 盲去卷积：无需知道退化函数，但难度最大
* 几何校正：包括空间变换与插值两部分
