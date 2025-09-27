## 第4章  图像增强复习笔记

### 4.1 概述与分类 (General Introduction and Classification)

* **目的**：改善图像视觉效果、便于边缘检测。
* **方法分类**：

  1. 空间域增强：点操作（灰度变换、直方图处理）、局部操作（平滑、锐化滤波）。
  2. 频域增强：DFT→滤波→IDFT。

---

### 4.2 对比度增强 (Contrast Enhancement)

#### 4.2.1 灰度变换（Point Operations）

* **通用形式**：$s = T(r)$，$r=f(x,y)$，$s=g(x,y)$。
* **线性变换**：
  $s = ar + b,\quad r\in[A,B]	o s\in[C,D],\;a=\frac{D-C}{B-A},\;b=\frac{BC-AD}{B-A}.$
* **分段线性变换**：定义阈值$r=a,b$上的多段斜率。
* **图像反转**：$s = L-1 - r$。
* **对数变换**：$s = c\ln(1+r)$。
* **幂律变换（伽马校正）**：$s = c\,r^\gamma$。
* **灰度切片**：保留某区间增强或映射到常数。
* **位面切片**：对二进制位提取

#### 4.2.2 直方图处理 (Histogram Processing)

* **直方图均衡化**：连续形式
  $s = T(r)=\int_0^r p_r(w)dw,\quad p_s(s)=1.$
  离散实现：
  $s_k = (L-1)\sum_{j=0}^k p_r(r_j),\quad p_r(r_j)=n_j/n.$
* **直方图规定化（匹配）**：先均衡化原直方图$r	o s$，再反变换目标直方图累积函数$s	o z$。

---

### 4.3 图像平滑 (Image Smoothing)

#### 4.3.1 噪声模型

* **高斯噪声**：$p(z)=\frac{1}{\sqrt{2\pi}\sigma}\exp[-(z-\mu)^2/(2\sigma^2)].$
* **椒盐噪声**：部分像素被设为高或低值。

#### 4.3.2 线性平滑滤波

* **空间域卷积**：
  $g(x,y)=\sum_{i=-M}^M\sum_{j=-N}^N w(i,j)f(x+i,y+j).$
* **均值滤波器**：$\frac{1}{mn}\mathbf{1}_{m\times n}$；高斯滤波器；加权中值。
* **边界处理**：零填充、镜像、阈值保留等。

#### 4.3.3 非线性平滑滤波（Order-Statistic）

* **中值滤波**：取邻域灰度中值。
* **最大/最小滤波**：消除椒盐噪声。
* **中点滤波**：$\frac{\max+\min}{2}$；**α-修剪均值**：去掉α个极值后取平均。

#### 4.3.4 频域低通滤波

* **理想低通 (ILPF)**：$H(u,v)=1$若$D\le D_0$，否则0。
* **Butterworth低通 (BLPF)**：$\displaystyle H=\frac{1}{1+[D/D_0]^{2n}}$.
* **Gaussian低通 (GLPF)**：$H=\exp[-D^2/(2D_0^2)].$

---

### 4.4 图像锐化 (Image Sharpening)

#### 4.4.1 引言


* **目的**：突出图像细节，便于边缘检测，对噪声不敏感。
* **方法分类**：

  1. 一阶导数算子（Prewitt、Sobel、Roberts、Kirsch、Robinson）
  2. 二阶导数算子（Laplacian、LoG）
  3. 频域高通滤波（理想高通、Butterworth、Gaussian、同态滤波）

---

#### 4.4.2 一阶导数（Gradient）

* **定义**：图像函数$f(x,y)$的一阶导数由梯度表示，

  $$
    \nabla f = \begin{bmatrix} f_x \\ f_y \end{bmatrix},
    \quad f_x=\frac{\partial f}{\partial x},\; f_y=\frac{\partial f}{\partial y}.
  $$
* **梯度幅值**：

  $$
    G = \|\nabla f\| \approx |f_x| + |f_y|.
  $$
* **梯度方向**：

  $$
    \alpha = \arctan\frac{f_y}{f_x}.
  $$
* **常见算子与掩模模板**：

  * **Roberts**: $G_x=\begin{smallmatrix}1 & 0\\0 & -1\end{smallmatrix}$, $G_y=\begin{smallmatrix}0 & 1\\-1 & 0\end{smallmatrix}$
  * **Prewitt**: $G_x=\begin{smallmatrix}-1&0&1\\-1&0&1\\-1&0&1\end{smallmatrix}$, $G_y=G_x^T$
  * **Sobel**: $G_x=\begin{smallmatrix}-1&0&1\\-2&0&2\\-1&0&1\end{smallmatrix}$, $G_y=G_x^T$
  * **Kirsch/Robinson**: 多方向模板，取最大响应

---

#### 4.4.3 二阶导数（Laplacian & LoG）

* **定义**：图像的Laplacian算子

  $$
    \nabla^2 f = f_{xx} + f_{yy}.
  $$
* **离散实现**：

  $$
    \nabla^2 f = f(x+1,y) + f(x-1,y) + f(x,y+1) + f(x,y-1) - 4f(x,y).
  $$

  对应掩模：

  $$
    \begin{bmatrix} 0 & 1 & 0\\ 1 & -4 & 1\\ 0 & 1 & 0 \end{bmatrix}.
  $$
* **Laplacian增强**：

  $$
    g(x,y) = f(x,y) + \lambda \nabla^2 f(x,y).
  $$
* **LoG (Laplacian of Gaussian)**：先进行高斯平滑后再做Laplacian：

  $$
    h(r)=\left(1-\frac{r^2}{2\sigma^2}\right)e^{-r^2/(2\sigma^2)}, \;r^2=x^2+y^2.
  $$

---

#### 4.4.4 频域高通滤波

* **理想高通 (IHPF)**：

  $$
    H(u,v)=\begin{cases}0,&D(u,v)\le D_0\\1,&D(u,v)>D_0\end{cases}.
  $$
* **Butterworth高通 (BHPF)**：

  $$
    H(u,v)=\frac{1}{1+[D_0/D(u,v)]^{2n}}.
  $$
* **Gaussian高通 (GHPF)**：

  $$
    H(u,v)=1-\exp\left(-\frac{D(u,v)^2}{2D_0^2}\right).
  $$
* **同态滤波 (Homomorphic)**：对数域中分离光照与反射，增强高频：

  * 变换：$z=\ln f = \ln i + \ln r$
  * 滤波后逆变换

---

### 4.5 彩色图像增强 (Color Image Enhancement)

#### 4.5.1 色彩模型基础

* **RGB**、**CMY(K)**、**HSI**等模型及其转换公式
* **HSI定义**：

  * 强度$I=(R+G+B)/3$
  * 饱和度$S=1-\frac{3}{R+G+B}\min(R,G,B)$
  * 色调$H=\begin{cases}\theta,&B\le G \\ 360^\circ-\theta,&B>G\end{cases}$，其中

    $$
      \theta=\arccos\frac{\tfrac{1}{2}[(R-G)+(R-B)]}{\sqrt{(R-G)^2+(R-B)(G-B)}}.
    $$

---

#### 4.5.2 伪彩色处理 (Pseudo Color)

* **强度切片**：映射灰度区间到颜色：

  $$
    f'(x,y)=c_k, \;{\rm if}\;f(x,y)\in V_k.
  $$
* **灰度到RGB映射**：独立对R、G、B通道做变换
* **多通道合成**：将多谱段图像合成假彩色图

---

#### 4.5.3 真彩色增强 (Full Color)

* **向量表示**：$\mathbf{c}(x,y)=[R,G,B]^T$
* **彩色空间变换**：在RGB或HSI空间中对各分量做点运算

  * **亮度变换**：在HSI仅修改I，在RGB中统一作用于R,G,B
  * **彩色反转**：$r_i'=1-r_i$
  * **直方图均衡**：HSI中仅对I分量均衡
  * **平滑与锐化**：可在各通道分别进行

---

#### 4.5.4 噪声处理

* RGB通道独立噪声与HSI空间中不同通道噪声特点
* 滤波（局部平均、中值滤波）可分别作用于各通道

---

*以上为第4章核心理论与常用公式汇总，供复习参考*
