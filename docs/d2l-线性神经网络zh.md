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

