d2l-线性神经网络

### 3.1 线性回归

**仿射变换（affine transformation）**：通过加权和对特征进行特征变化进行线性变换，并通过偏置项来进行平移（translation）。

对于模型而言，需要两个东西，

**（1）一种模型质量的度量方式； （2）一种能够更新模型以提高模型预测质量的方法。**

**解析解（analytical solution)**：线性回归的解可以用一个公式简单的表达，称为解析解。解析解指的是通过严格公式推导所求得的解。

**随机梯度下降（gradient descent)**：最简单的办法是计算损失函数关于模型参数的导数，但实际执行过程可能速度比较慢。因而存在变体（minibatch stochastic gradient descent）。

**学习率（learning rate）** 

![image-20230607213849344](/Users/songtingyu/Library/Application Support/typora-user-images/image-20230607213849344.png)1.
$$
y = \sum_i(x_i- b)^2
$$

$$
y' = \sum_i(2x_i - 2b) = 2\sum_ix_i - 2nb 
$$

$$
b = \frac{1}{n} \sum_ix_i
$$

2. 平方误差的线性回归优化问题的解析解

   (1-3) 用矩阵和向量表示法写出优化问题

   $Y =
   \begin{pmatrix} y_1\\ y_2 \\ y_3 \\ \vdots \\ y_n\end{pmatrix}
   $, $\omega = 
   \begin{pmatrix} \omega_1\\ \omega_2 \\ \omega_3 \\ \vdots \\ \omega_n\end{pmatrix}
   $, $X = \begin{pmatrix} x_1,x_2,x_3, \dots, x_n \end{pmatrix}$  ,$Y = X \omega + b$

   又将偏置值加入权重，所以X变为$X_1 = \begin{bmatrix}X,1\end{bmatrix}$,$\omega_1 = \begin{bmatrix} \omega \\ b \end{bmatrix}$

   $Y =  X_1\omega_1  + \epsilon$，$\epsilon$为残差，$\epsilon = \frac{1}{2}(y-\hat{y})^2$ ($\frac{1}{2}是为了求导后系数为1$)

   找到合适的$\omega_1$，使得，${\begin{Vmatrix} \epsilon \end{Vmatrix}}_2$尽量小。

   $L(X_1,y,\omega) = $

   (2) 计算损失对$\omega$的梯度

   (3) 找到解析解

   (4) 什么时候使用比随机梯度下降好，

3. 绝对值的驻点是无法求导的。



### 3.2 面向对象的设计思想实现线性神经网络（Object-Oriented Design for Implementation）

主要是学习架构。包含3个对象，`Module`包含模型、损失函数以及优化的方法，`DataModule`提供了数据加载的相关方法，而`Trainr`类包括了这两个因素。

### 3.3 构造回归的数据（Synthetic Regression Data）

主要讲了如何构造以及加载数据，

```python
@d2l.add_to_class(d2l.DataModule)  #@save
def get_tensorloader(self, tensors, train, indices=slice(0, None)):
    tensors = tuple(a[indices] for a in tensors)
    dataset = torch.utils.data.TensorDataset(*tensors)
    # torch.utils.data.DataLoader represents a Python iterable over a dataset
    return torch.utils.data.DataLoader(dataset, self.batch_size,
                                       shuffle=train)

@d2l.add_to_class(SyntheticRegressionData)  #@save
def get_dataloader(self, train):
    i = slice(0, self.num_train) if train else slice(self.num_train, None)
    return self.get_tensorloader((self.X, self.y), train, i)
```

用了一些torch的工具，感觉增加了理解难度。可能在之后使用的时候才有感觉。

查了一下需要使用iter的原因，是因为iter这一类更加节省空间，并且按需生成数据，在处理大数据时非常有用。

