## 2 Preliminaries

### 2.1 Data Manipulation

```python
import torch
x = torch.arange(12,dtype=torch.float32) # 生成1维
x.numel() # 获取大小
x.shape() # 形状
X = x.reshape(3,4) # 改变形状
X = x.reshape(-1,4) # 使用-1时，代表该值自动推理。
# tensor 生成
torch.zeros((2,3,4)) # 构建(2,3,4)形状的tensor
torch.ones((2,3,4))
torch.randn((3,4))   # 生成EX=0，DX=1的高斯分布矩阵
torch.randn(3,4)		 # 与上一句效果一样
# 切片与索引
X[-1] # 选择相对最后一个的位置
X[1:3] # first but not last, 最后一个不选，选的是1，2行
X[:2,:] = 12 # 批量赋值
# Operations
# Elementwise operations 对每一个元素进行一个操作，也叫做单目运算符
# 还有双目运算符，Concatenate 可以在x-axis和y-axis上进行
X = torch.arange(12,dtype=torch.float32).reshape(3,4)
Y = torch.tensor([[2.0,1,4,3],[1,2,3,4],[4,3,2,1]])
torch.cat((X,Y),dim=0), torch.cat((X,Y),dim=1)
# Broadcasting
# 当两个参加运算的向量或者矩阵形状不匹配时，运算时会触发broadcast机制
# (1) 沿着长度为1的轴 扩充其中的1个或者两个 (2)之后进行elementwise的操作
a = torch.arange(3).reshape(3,1)
b = torch.arange(2).reshape(1,2)
a + b
# a = [[0],       b = [[0,1]]
#			 [1],
#	     [2]] 
# 扩充之后变为
# a = [[0,0],       b = [[0,1],
#			 [1,1],						 [0,1],
#	     [2,2]] 					 [0,1]]
```

接下来讲了一些可以通过切片等方式来节省内存。另一方面是pytorch中的tensor可以和numpy中的ndarray进行相互转化。

### 2.3 Linear Algebra

**张量** 描述具有任意数量轴的n维数组的通用方法。

**Hadamard积（数学符号⊙）**
$$
\mathbf{A} \odot \mathbf{B} =

\begin{bmatrix}

​    a_{11}  b_{11} & a_{12}  b_{12} & \dots  & a_{1n}  b_{1n} \\

​    a_{21}  b_{21} & a_{22}  b_{22} & \dots  & a_{2n}  b_{2n} \\

​    \vdots & \vdots & \ddots & \vdots \\

​    a_{m1}  b_{m1} & a_{m2}  b_{m2} & \dots  & a_{mn}  b_{mn}

\end{bmatrix}.
$$
将张量乘以一个标量不会改变张量的形状。

**降维** 调用求和函数会使得张量降维

```python
x = torch.arange(4,dtype=torch.float32)
x, x.sum()
A = torch.arange(20,dtype=torch.float32).reshape(5,4)
A_sum_axis1 = A.sum(axis=1) # 输入轴1的维数在输出形状中消失
A_sum_axis0 = A.sum(axis=0) # 输入轴0的维数在输出形状中消失 
A.sum(axis=[0,1]) # 延横轴和纵轴进行求和
A.sum() / A.numel() # 求平均值
```

**非降维求和**

```python
A = torch.arange(20,dtype=torch.float32).reshape(5,4)
sum_A1 = A.sum(axis = 1) # 1
sum_A2 = A.sum(axis = 1,keepdims=True) # 2
A / sum_A
A.cumsum(axis=0) # 延某一轴累计，左式把
```

1 的结果是 $\begin{bmatrix} 6,22,38,54,70 \end{bmatrix}$

2 的结果是 
$$
\begin{bmatrix}
[6],
[22],
[38],
[54],
[70]
\end{bmatrix}
$$

---

**点积、矩阵-向量积、矩阵-矩阵乘法**

**点积**

```python
x = torch.arange(4,dytpe=torch.float32)
y = torch.ones(4,dtype=torch.float32)
x,y = torch.dot(x,y)
```

比较有用的是，当y的和为1时，点积表示加权平均。

**矩阵-向量积**

```python
A = torch.arange(20,dtype=torch.float32).reshape(5,4)
x = torch.arange(4,dytpe=torch.float32)
torch.mv(A,x)
```

**矩阵-矩阵乘法**

```python
B = torch.ones(4,3)
torch.mm(A,B)
```

---

**范数**

表示一个向量有多大（Size）。概念不涉及维度，而是分量的大小。

向量范数时将向量映射到标量的函数f。

**第一个性质**是：如果我们按常数因子$\alpha$缩放向量的所有元素，其范数也会按相同常数因子的**绝对值**缩放：

$$f(\alpha \mathbf{x}) = |\alpha| f(\mathbf{x}).$$

**第二个性质**是熟悉的三角不等式:

$$f(\mathbf{x} + \mathbf{y}) \leq f(\mathbf{x}) + f(\mathbf{y}).$$

**第三个性质**简单地说范数必须是非负的:

$$f(\mathbf{x}) \geq 0.$$

欧几里得距离是一个L2范数。

$L_2$范数和$L_1$范数都是更一般的$L_p$范数的特例：

$$\|\mathbf{x}\|*_p = \left(\sum_*{i=1}^n \left|x_i \right|^p \right)^{1/p}.$$

类似于向量的$L_2$范数，[**矩阵**]$\mathbf{X} \in \mathbb{R}^{m \times n}$(**的Frobenius范数（Frobenius norm）是矩阵元素平方和的平方根：**)

(**$$\|\mathbf{X}\|*_F = \sqrt{\sum_*{i=1}^m \sum*_{j=1}^n x_*{ij}^2}.$$**)