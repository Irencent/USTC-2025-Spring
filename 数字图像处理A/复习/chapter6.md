《数字图像处理》第六章 图像重建 - 复习笔记

---

# 第6章 图像重建（Image Reconstruction）

## 一、引言（6.1 Introduction）

### 6.1.1 CT简史

* 1895：伦琴发现X射线。
* 1964：Cormack从数学上证明X射线吸收与组织密度有关。
* 1972：Hounsfield发明第一台CT机。

### 6.1.2 投影的物理基础

* **X射线强度衰减模型**：

  * 块体组织：$I = I_0 e^{-\mu L}$
  * 连续分布：$I = I_0 e^{-\int_L \mu(x, y) , dl}$
  * 对数形式：$\ln(\frac{I_0}{I}) = \int_L \mu(x, y) , dl$

### 6.1.3 重建目标

* 根据多个投影重建二维衰减分布$\mu(x, y)$

### 6.1.4 重建方法分类

* 傅里叶反演法（频域）
* 卷积反投影法（空间域）
* 有限级数展开法（空间域）

---

## 二、傅里叶反演重建法（6.2 Reconstruction by Fourier Inversion）

### 6.2.1 投影的数学表达

* x轴方向投影：$p(x) = \int_{-\infty}^{\infty} f(x, y) , dy$
* 任意方向$\theta$：$p_\theta(t) = \int f(x, y) , ds$
* 坐标变换：

  * $t = x\cos\theta + y\sin\theta$
  * $s = -x\sin\theta + y\cos\theta$

### 6.2.2 傅里叶切片定理（Fourier Slice Theorem）

* $F(u, v)$是$f(x, y)$的二维傅里叶变换。
* $S_\theta(\omega)$是投影$p_\theta(t)$的一维傅里叶变换。
* 定理内容：$S_\theta(\omega) = F(\omega \cos\theta, \omega \sin\theta)$，即$S_\theta(\omega)$是$F(u,v)$的切片。

### 6.2.3 傅里叶反演重建步骤

1. 对每个角度$\theta_m$计算投影的DFT：$S_{\theta_m}(\omega)$
2. 将所有$S_{\theta}(\omega)$合成为$F(\omega, \theta)$
3. 插值到笛卡尔网格上
4. 执行二维反傅里叶变换（IDFT）得到$f(x, y)$

---

## 三、卷积反投影法（6.3 Reconstruction by Convolution and Backprojection）

### 6.3.1 平行束重建推导

* 利用傅里叶反演推导得：
  $f(x, y) = \int_0^{\pi} \int_{-\infty}^{\infty} S_\theta(\omega) e^{j2\pi \omega (x \cos\theta + y \sin\theta)} , d\omega , d\theta$
* 令$t = x \cos\theta + y \sin\theta$，再变换得：
  $f(x, y) = \int_0^{\pi} p_\theta(t) * h(t) , d\theta$

### 6.3.2 滤波器$h(t)$

* $h(t)$为$S_\theta(\omega)$的傅里叶逆变换
* 理想滤波器：
  $h(t) = \frac{1}{2\tau} \left[ \frac{\sin^2(\pi t / \tau)}{(\pi t / \tau)^2} \right]$

### 6.3.3 算法步骤

1. 对每个角度的投影$p_\theta(t)$进行滤波，得$p'_\theta(t)$
2. 对$p'*\theta(t)$做反投影积分：
   $f(x, y) = \int_0^{\pi} p'*\theta(t) , d\theta$

### 6.3.4 常见问题

* **角度采样不足**：会导致重建图像锯齿/条纹伪影（aliasing）
* **径向采样不足**：由高频细节（如骨骼）引起伪影
* **运动伪影**：患者呼吸、心跳等引起数据失真

---

## 四、有限级数展开法（6.4 Finite Series-Expansion）

### 6.4.1 离散投影表达

* 将二维图像向量化为$f = [f_0, f_1, ..., f_{N-1}]$
* 第$i$条射线的投影为：$p_i = \sum_{j=0}^{N-1} w_{ij} f_j$
* $w_{ij}$为投影矩阵，表示射线$i$是否穿过像素$j$

### 6.4.2 迭代求解

* 看作线性方程组$Wf = p$
* 迭代更新公式：
  $f_j^{(k+1)} = f_j^{(k)} + \frac{p_i - \sum w_{ij} f_j^{(k)}}{\sum w_{ij}^2} w_{ij}$

### 6.4.3 算法步骤

1. 初始化$f^{(0)}$
2. 按投影方程调整$f^{(k)} \rightarrow f^{(k+1)}$
3. 若收敛（调整小于阈值$\delta$），则停止

---

## 五、作业题（Homework）

1. 简述CT发明过程（伦琴、Cormack、Hounsfield）
2. 证明投影定理（Fourier Slice Theorem）
