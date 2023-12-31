d2l-线性神经网络en.md

### 3.1 线性回归

**仿射变换（affine transformation）**：通过加权和对特征进行特征变化进行线性变换，并通过偏置项来进行平移（translation）。

对于模型而言，需要两个东西，

**（1）一种模型质量的度量方式； （2）一种能够更新模型以提高模型预测质量的方法。**

**解析解（analytical solution)**：线性回归的解可以用一个公式简单的表达，称为解析解。解析解指的是通过严格公式推导所求得的解。

**随机梯度下降（gradient descent)**：最简单的办法是计算损失函数关于模型参数的导数，但实际执行过程可能速度比较慢。因而存在变体（minibatch stochastic gradient descent）。

**学习率（learning rate）** 

1.
$$
y = \sum_i(x_i- b)^2
$$

$$
y' = \sum_i(2x_i - 2b) = 2\sum_ix_i - 2nb 
$$

$$
b = \frac{1}{n} \sum_ix_i
$$

1. 平方误差的线性回归优化问题的解析解

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

2. 绝对值的驻点是无法求导的。



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

### 3.4 Linear Regression from scratch
![](../pic/3-4.png)

1. What would happen if we were to initialize the weights to zero. Would the algorithm still work? What if we initialized the parameters with variance 1000 rather than 0.01 ?
    > Yes, it still work. Then the predicted result will larger than real. If we increase the epochs number, than we will get a more precise result.

2. Assume that you are Georg Simon Ohm trying to come up with a model for resistors that relate voltage and current. Can you use automatic differentiation to learn the parameters of your model?
    > $$I = V / R$$

3. Derivate it.
   $$
    
   $$

4. What are the problems you might encounter if you wanted to compute the second derivatives of the loss? How would you fix them?
    > * Second derivatives will be zero.
    > * hello world

5. Why is the reshape method needed in the loss function?

6. Experiment using different learning rates to find out how quickly the loss function value drops. Can you reduce the error by increasing the number of epochs of training?
   > The loss will get down faster. But we cant reach minima in 3 epochs. As Figure 1 shows.
   > ![lr=0.05,epochs=3](../pic/en-3-4-1.png) 
   > <center>Fig 1. lr = 0.05 epochs = 3</center>

   >![lr=0.05,epochs=5](../pic/en-3-4-2.png)
   > <center>Fig 2. lr = 0.05 epochs = 5</center>
   
7. If the number of examples cannot be divided by the batch size, what happens to data_iter at the end of an epoch?
    > We have 2 choices, one is to not use these examples. The other is to use the remain with indices[i:num_examples]

8. Try implementing a different loss function, such as the absolute value loss (y_hat - d2l.reshape(y, y_hat.shape)).abs().sum().
    > 1. Check what happens for regular data.
    > We get a bigger loss.
    > ![abs loss](../pic/en-3-4-3.png)
    > <center>Fig 3. abs loss</center>
    > Check whether there is a difference in behavior if you actively perturb some entries of y, such as $y_5=10000$.
    > I cannot draw some conclution.

    > ![](../pic/en-3-4-4.png)
    > <center>Fig 4. y5 = 10000</center>
    > Can you think of a cheap solution for combining the best aspects of squared loss and absolute value loss? Hint: how can you avoid really large gradient values?
    > I divide the abs result with a constant or maybe we can just sqrt it.(I dont know whether it's right.)

9.  Why do we need to reshuffle the dataset? Can you design a case where a maliciously dataset would break the optimization algorithm otherwise?
    > Because in examples, we have examples that are arranged together for some order. They may be the same value or following the rule that each add one from previous one. So after shuffle, we can get a more accurate variance and lower the bias.

### 3.5 Concise Implementation of Linear Regression

* Lazy Linear
* torch operations

1. How would you need to change the learning rate if you replace the aggregate loss over the minibatch with an average over the loss on the minibatch?
   > It's asking about how do you use MSELoss( none | mean | sum ).
2. Review the framework documentation to see which loss functions are provided. In particular, replace the squared loss with Huber’s robust loss function. That is, use the loss function
    
<center> 

![](../pic/en-3-5-1.png) 

</center>

    > torch.nn.SmoothLoss()


3. How do you access the gradient of the weights of the model?
   > In ./train.py call W.grad is None. Oops.
   > So I add a method `get_w_grad` in model.py

### 3.6 Generalization
Mainly talks about fit and overfit.

> In the **standard supervised learning** setting, we assume that the training data and the test data are drawn **independently** from **identical** distributions. 

3.6.2 talks about underfitting and overfitting, relating to polynomial curve, datasets size and model selection.

**3.6.3 cross validation** with no enough validation set, we divide training sets to k parts. Then we train and do validation k times. Each time, we do train on k-1 sets and do validation on the remain set.

> 1. Use validation sets (or -fold cross-validation) for model selection;
> 2. More complex models often require more data;
> 3. Relevant notions of complexity include both the number of parameters and the range of values that they are allowed to take;
> 4. Keeping all else equal, more data almost always leads to better generalization;
> 5. This entire talk of generalization is all predicated on the IID assumption. If we relax this assumption, allowing for distributions to shift between the train and testing periods, then we cannot say anything about generalization absent a further (perhaps milder) assumption.

The questions in 3.6 is about some tricks in training and validation.

### 3.7 Weight Decay
<center> lambda = 0</center>

![](../pic/en-3-7-1.png)

<center> lambda = 3</center>

![](../pic/en-3-7-2.png)

可以看到在使用权重衰减后，验证集收敛的更快。

**3.7习题**
1. Experiment with the value of in the estimation problem in this section. Plot training and validation accuracy as a function of . What do you observe?
   > ![](../pic/en-3-7-3.png)
   > lambda 数值并不是越大越好，取到某些数值如4、6可以获得较为符合的结果。

2. Use a validation set to find the optimal value of $\lambda$. Is it really the optimal value? Does this matter?
   > 这题给我整不会了。

3. What would the update equations look like if instead of ${\| w \|}^2$ we used $\Sum_i|w_i}$ as our penalty of choice (regularization)?
   > ```python
   > def l1_penalty(w):
   >    return w.sum().abs()
   > ```

4. We know that $\| w \| = 2w^Tw$. Can you find a similar equation for matrices (see the Frobenius norm in Section 2.3.11)?
   > $$ {\| X \|}_F = \sqrt {\sum^m_{i=1} \sum^n_{j=1} x_{ij}^2 }$$
   > ```python
   > torch.norm(torch.ones(4,9))
   > ```

5. Review the relationship between training error and generalization error. In addition to weight decay, increased training, and the use of a model of suitable complexity, what other ways can you think of to deal with overfitting?
   > 问还有什么办法解决overfitting，可以通过dropout或者激活函数降低过拟合的几率。

6. In Bayesian statistics we use the product of prior and likelihood to arrive at a posterior via $ P(w|x) \propto P(x|w)P(w) $. How can you identify P(w) with regularization?
   > I don't know the answer.