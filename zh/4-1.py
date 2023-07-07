import torch 
from d2l import torch as d2l

x = torch.arange(-8.0,8.0,0.1,requires_grad=True)
# y = torch.relu(x)
# 绘制ReLU曲线
# d2l.plot(x.detach(),y.detach(),'x','relu_x',figsize=(5,2.5))

# 绘制ReLU函数导数

# 只有计算标量时，输出的梯度会被自动创建
# 比如 
# y = y.sum()
# y.backward()才可以不显示指定梯度起点
# y.backward(torch.ones_like(x),retain_graph=True)

# d2l.plot(x.detach(),x.grad,'x','grad of relu',figsize = (5,2.5))

y = torch.sigmoid(x)
d2l.plot(x.detach(),y.detach(),'x','sigmoid_x',figsize=(5,2.5))
d2l.plt.show()