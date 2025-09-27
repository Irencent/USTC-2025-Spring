**Chapter 3: Image Transforms — 复习笔记**

---

## 3.1 一般介绍与分类

**定义与矩阵表达**

* 线性变换：若 $X$ 为 $N\times1$ 向量，$T$ 为 $N\times N$ 矩阵，则
  $Y=TX,\quad y_i=\sum_{j=0}^{N-1} t_{ij}x_j.$
* 可逆性：当 $\mathrm{rank}(T)=N$ 时可逆；
  $T^{-1}\,\exists,\quad T^{-1}T=I.$
* 单位变换 (Unitary)：若 $T^{-1}=T^*$；
* 正交变换 (Orthogonal)：若 $T^{-1}=T^T$.

**分类**

* **可分离(separable)**：
  $\Phi(i,j;m,n)=T_r(i,m),T_c(j,n)$
* **对称(symmetric)**：
  $T_r=T_c$
* **统计变换**：Hotelling

| 类别      | 代表变换                         | 备注           |
| ------- | ---------------------------- | ------------ |
| 正弦/余弦变换 | DFT, DCT                     | 分析频域特性       |
| 其它可分离变换 | Walsh, Hadamard, Haar, Slant | 简化硬件实现       |
| 统计变换    | Hotelling                    | 基于协方差矩阵的正交分解 |

---

## 3.2 傅里叶变换及其性质

### 3.2.1 定义

**连续傅里叶变换 (CFT)**
$F(u)=\int_{-\infty}^{\infty} f(x)e^{-j2\pi ux}\,dx, \quad f(x)=\int_{-\infty}^{\infty} F(u)e^{j2\pi ux}\,du.$

**离散傅里叶变换 (DFT)**
1D:
$F(u)=\sum_{x=0}^{N-1} f(x)e^{-j2\pi ux/N}, \quad f(x)=\frac1N\sum_{u=0}^{N-1}F(u)e^{j2\pi ux/N}.$
2D:
$F(u,v)=\sum_{x=0}^{M-1}\sum_{y=0}^{N-1}f(x,y)e^{-j2\pi(ux/M+vy/N)},$
$f(x,y)=\frac1{MN}\sum_{u=0}^{M-1}\sum_{v=0}^{N-1}F(u,v)e^{j2\pi(ux/M+vy/N)}.$

**谱与相位**

* 幅度谱: $|F(u,v)|=\sqrt{R^2+I^2}$
* 相位谱: $\Phi(u,v)=\arctan\frac{I}{R}$
* 功率谱: $P(u,v)=|F(u,v)|^2$

### 3.2.2 性质

1. **可分离性**：先沿行再沿列
2. **周期性 & 共轭对称**：
   $F(u+M,v)=F(u,v+N)=F(u,v),\quad F^*(u,v)=F(-u,-v).$
3. **平移**：空间域平移只影响相位
   $f(x-x_0,y-y_0)\leftrightarrow F(u,v)e^{-j2\pi(ux_0/M+vy_0/N)}.$
4. **旋转**：
   $f(r,\theta+\alpha)\leftrightarrow F(\rho,\phi+\alpha).$
5. **对率**：
   $f(ax,by)\leftrightarrow \frac1{|ab|}F\bigl(u/a,v/b\bigr).$
6. **线性与分布性**：
   $a f+ b g \leftrightarrow aF+bG.$
7. **平均值**：
   $\bar f=\frac1{MN}\sum f(x,y)=\frac1{MN}F(0,0).$
8. **卷积定理**：
   $f* g \leftrightarrow F\cdot G,\quad f\cdot g\leftrightarrow F*G.$

**频谱显示**

* 对数压缩：$D(u,v)=\log\bigl(1+|F(u,v)|\bigr)$
* 均衡拉伸：
  $D(u,v)=\begin{cases}F(u,v),&F\le100,\\100+(F-100)\frac{155}{155},&F>100.\end{cases}$

---

## 3.3 其它可分离变换

### 3.3.1 离散余弦变换 (DCT)

1D DCT:
$C(u)=\sum_{x=0}^{N-1}a(u)f(x)\cos\bigl[\frac{\pi(2x+1)u}{2N}\bigr],$
$f(x)=\sum_{u=0}^{N-1}a(u)C(u)\cos\bigl[\frac{\pi(2x+1)u}{2N}\bigr],$
$a(u)=\begin{cases}\sqrt{1/N},&u=0,\\\sqrt{2/N},&u\ge1.\end{cases}$
2D DCT:
$C(u,v)=\sum_{x,y=0}^{N-1}a(u)a(v)f(x,y)\cos[\tfrac{(2x+1)u\pi}{2N}]\cos[\tfrac{(2y+1)v\pi}{2N}].$

### 3.3.2 Walsh / Hadamard

* 基于二进制位互异或运算，核可分离且对称。
* 1D:
  $W(u)=\sum_{x=0}^{N-1}f(x)(-1)^{\sum_i b_i(x)b_i(u)},$
  $f(x)=\frac1N\sum_uW(u)(-1)^{\sum_i b_i(x)b_i(u)}.$

### 3.3.3 哈达玛变换 (Hadamard)

* 特殊二值矩阵生成：
  $H_2=\begin{pmatrix}1&1\\1&-1\end{pmatrix},\quad H_N=H_2\otimes H_{N/2}.$
* 正交无归一化：
  $H(u)=\sum_{x=0}^{N-1}f(x)h_{u,x},\quad f(x)=\frac1N\sum_uH(u)h_{u,x}.$

### 3.3.4 哈尔变换 (Haar)

### 5. 二维 Haar 变换


一级 Haar 变换可写成矩阵形式

$$
H_2 = \frac1{\sqrt2}
\begin{pmatrix}
 1 &  1\\
 1 & -1
\end{pmatrix},
$$

一般长度 $N$ 的变换矩阵 $H_N$ 可递归用 Kronecker 积构造：

$$
H_{2N} 
= \frac{1}{\sqrt2}
\begin{pmatrix}
  H_N &  H_N\\
  H_N & -H_N
\end{pmatrix}
= H_2 \otimes H_N.
$$

对信号向量 $f$ 应用 $H_N$ 即得一次分解。

对于 $M\times N$ 图像 $f(x,y)$，利用可分离性：

1. 对每一行应用 1D Haar 变换；
2. 再对每一列应用 1D Haar 变换。

等价于

$$
F = H_M \;f\; H_N^T,
$$

其中 $H_M$ 作用于行，$H_N$ 作用于列。

* 分段常数基函数：在区间 $[0,1]$ 上分级分段
* 1D Haar:
  $H_k(z)=\begin{cases}1,&k=0,\\2^{p/2},&z\in[\tfrac{q-1}{2^p},\tfrac{q-0.5}{2^p}),\\-2^{p/2},&z\in[\tfrac{q-0.5}{2^p},\tfrac{q}{2^p}),\\0,&\text{otherwise.}\end{cases}$

---


## 3.4 Hotelling

### 3.4.1 定义与目的

Hotelling 变换（又称主成分分析 PCA）是一种统计变换，通过对多维数据集进行正交分解，找出数据集中方差最大的正交方向，从而实现降维、去相关和特征提取。

### 3.4.2 数据中心化

给定样本集 $\{\mathbf{x}_k\}_{k=1}^K$，其中 $\mathbf{x}_k\in\mathbb{R}^N$，首先计算均值向量：

$$
\bar{\mathbf{x}} = \frac{1}{K}\sum_{k=1}^K \mathbf{x}_k
$$

然后对每个样本去中心化：

$$
\tilde{\mathbf{x}}_k = \mathbf{x}_k - \bar{\mathbf{x}}.
$$

### 3.4.3 协方差矩阵

中心化后数据的协方差矩阵定义为：

$$
\mathbf{C}
= \frac{1}{K}\sum_{k=1}^K \tilde{\mathbf{x}}_k\,\tilde{\mathbf{x}}_k^T
= E\bigl[\tilde{\mathbf{x}}\tilde{\mathbf{x}}^T\bigr].
$$

此矩阵对称且正半定，度量各维度间的线性相关性。

### 3.4.4 特征分解

对协方差矩阵进行特征分解：

$$
\mathbf{C}\,\boldsymbol{\Phi}
= \boldsymbol{\Phi}\,\boldsymbol{\Lambda},
$$

其中

* $\boldsymbol{\Phi} = [\phi_1,\phi_2,\dots,\phi_N]$ 为按方差（特征值）大小降序排列的特征向量矩阵，
* $\boldsymbol{\Lambda} = \mathrm{diag}(\lambda_1,\lambda_2,\dots,\lambda_N)$ 为对应的特征值对角矩阵。

### 3.4.5 变换与逆变换

* **Hotelling 变换（投影）**：将中心化向量投影到特征向量空间，得到主成分向量

  $$
  \mathbf{y}_k = \boldsymbol{\Phi}^T\,\tilde{\mathbf{x}}_k,
  \quad
  y_{k,i} = \phi_i^T(\mathbf{x}_k - \bar{\mathbf{x}}).
  $$
* **逆变换（重构）**：

  $$
  \hat{\mathbf{x}}_k = \boldsymbol{\Phi}\,\mathbf{y}_k + \bar{\mathbf{x}}.
  $$

若只保留前 $M<N$ 个主成分，则可以近似重构：

$$
\hat{\mathbf{x}}_k \approx
\sum_{i=1}^M y_{k,i}\,\phi_i \;+\;\bar{\mathbf{x}}.
$$

### 3.4.6 特点与应用

* **去相关**：变换后的各分量彼此正交，协方差为零；
* **降维**：仅保留前 $M$ 个方差最大的分量，可有效压缩数据；
* **应用场景**：图像压缩、特征提取、噪声抑制、数据可视化等。


