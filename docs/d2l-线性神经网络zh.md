### 3.2 线性回归的从零开始实现
代码问题，使用如下函数得到的最终结果较为符合期望，误差为小数点后4位。是因为如果使用l.sum()那么保存了每一个变量对于梯度的贡献大小，而l.mean()考虑的是综合的变量，对每一个变量的梯度之和求平均，导致梯度细节消失。通常在小样本量时，使用l.sum.backward()。而在较大样本量时，并且更加关注平稳性，可以使用l.mean.backward()。
```python
def sgd(params,lr,batch_size):
    with torch.no_grad():
        # 这里传递的是引用
        for param in params:
            param -= lr * param.grad  / batch_size
            param.grad.zero_()

for epoch in range(num_epochs):
    for X,y in data_iter(batch_size,features,labels):
        l = loss(net(X,w,b),y)
        l.sum().backward()
        sgd([w,b],lr,batch_size)
    with torch.no_grad():
        train_l = loss(net(features,w,b),labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
```
使用如下代码，得到的误差为小数点后3位。
```python
def sgd(params,lr,batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad
            param.grad.zero_()

for epoch in range(num_epochs):
    for X,y in data_iter(batch_size,features,labels):
        l = loss(net(X,w,b),y)
        l.mean().backward()
        sgd([w,b],lr,batch_size)
    with torch.no_grad():
        train_l = loss(net(features,w,b),labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
```

### 3-3 线性回归的简洁实现
主要是讲了人工智能框架的使用方法，看完英文版的3-4这里非常好理解。 或者不看其实也挺好理解的。

### 3-4 Softmax
one-hot 编码
softmax 函数能够将未规范化的预测变换为非负数并且总和为1.
$$
\hat{y} = softmax(o) , \hat{y}_j = \frac{exp(o_j)}{\sum_k {exp(o_k)}}
$$

对数似然
$$
P(Y|X) = \Pi^{n}_{i=1} P(y^{i} | x^{i})
$$

$$
-logP(\bold{Y}|\bold{X}) = \sum^n_{i=1} -logP(y^{(i)} | x^{(i)}) = \sum^n_{i=1} l(y^{(i)},\hat{y}^{(i)})
$$

损失函数（也称为交叉熵损失）。通过 这个损失函数对 $\hat{y_j}$ 求导其实就很容易理解为什么损失函数定义成这样了。求导之后的结果是 $soft\_max(\hat(y_j))- y_j$ ，也就是真实值和理论值的差距。通过最小化这个函数就可以实现目的。
$$
l(y,\hat{y}) = -\sum^{q}_{j=1} y_j log\hat{y_j}
$$

1. 指数族与softmax之间的关系
    1. 计算softmax交叉熵损失$l(y,\hat{y})$的二阶导数
    > $$l' = \frac{exp(o^j)}{\sum^n_{k=1} exp(o_k)} - y_i = softmax(o_j) - y_j$$
    > $$l^{''} =\frac{-(e^{o_j})^2 + e^{o_j} * {\sum^n_{k=1} e ^ {o_k}}} {(\sum^n_{k=1} e^{o_k}) ^ 2}$$
    > $$ l^{''} = - {softmax(o_j)}^ 2 + softmax(o_j) $$
    2. 计算softmax(o)给出的分布方差
    > $$D(X) = E(X^2) - E^2(X) $$
    > $$ E(o) = \sum^n_{k=1} \frac{o_j * e^{o_j}}{\sum^n_{k=1} e^{o_k} } $$
    > $$ E(o^2) = \sum^n_{k=1} \frac{{o_j}^2 * e^{o_j}}{\sum^n_{k=1} e^{o_k} } $$
    > $$ D(o) = E(o^2) - E^2(o) $$
    > $$ D(o) = \sum^n_{k=1} \frac{{o_j}^2 * e^{o_j}}{\sum^n_{k=1} e^{o_k} } - (\sum^n_{k=1} \frac{o_j * e^{o_j}}{\sum^n_{k=1} e^{o_k} })^2 $$
    > $$ D(o) = \sum^n_{k=1} o_k softmax(o_k) - \sum^n_{i=1}\sum^n_{j=1}(o_i o_j softmax(o_j) * softmax(o_i))$$ 
    > $$ D(o) = \sum^n_{i=1} \sum^n_{j=1(i!=j)} (o_j^2-o_io_j) \frac{e^{o_i}e^{o_j}}{(\sum^n_{k=1}e^{o_k})^2} $$
    > $$ D(o) = \sum^n_{i=1} \sum^n_{j=1(i!=j)} (o_j^2-o_io_j) \frac{\partial l}{\partial o_i \partial o_j}$$

2. 三个类发生的概率相等，概率向量为$(\frac{1}{3},\frac{1}{3},\frac{1}{3})$.
   1. 为他设置二进制代码，有什么问题。
    > 浪费了1位，二进制代码需要2位，可以表示4个情况，但是有一种情况被浪费了。另一种是考虑海明距离的问题，如果00-01-11，00和11显然差距比00-01以及01-11更大。
   2. 两个独立的观察结果会发生什么，联合编码n个值怎么办。
    > One-hot 编码
3. 真正的softmax被定义为$RealSoftMax(a,b) = log(exp(a)+exp(b))$
   1. 证明 RealSoftMax(a,b) > max(a,b)
    > 两边取e次幂，
    > 得到$exp(a) + exp(b)$ 以及 $exp(max(a,b))$
    > $又\because exp(a) > 0,exp(b) > 0, exp(max(a,b)) = exp(a) 或 exp(b)$
    > 可以得 $exp(a) + exp(b) > exp(max(a,b)) $
    > 从而有 $ RealSoftMax(a,b) > max(a,b)$
   2. $\lambda^{-1} RealSoftMax(\lambda a,\lambda b) > max(a,b)$成立，前提是 $\lambda > 0$
   > 反证法
   3. 证明对于$\lambda \to \inf$，有$\lambda^{-1}RealSoftMax(\lambda a,\lambda b) \to max(a,b)$
   > $${\lambda^{-1} log(exp(\lambda a)+exp(\lambda b))} \to {max(a,b)}$$
   > $$ a \ge b $$
   > when $a > b$
   > $$exp(\lambda a) >> exp(\lambda b) $$
   > $$ {\lambda^{-1} log(exp(\lambda a)+exp(\lambda b))} \to {\lambda^{-1} log(exp(\lambda a))} \to a = max(a,b) $$

   4. softmin
   > softmax(-x)
   5. 扩展到两个以上数字
   > $$ RealSoftMax = log(\sum^n_{i=1} exp(x_i)) $$ 