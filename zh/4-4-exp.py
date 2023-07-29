import math
import numpy as np
import torch
from torch import nn

from d2l import torch as d2l

from matplotlib import pyplot as plt

import sys
sys.path.append('..')
from dl import torch_zh as dl

def evaluate_loss(net,data_iter,loss):
    metric = dl.Accumulator(2)
    for X,y in data_iter:
        out = net(X)
        y = y.reshape(out.shape)
        l = loss(out,y)
        metric.add(l.sum(),l.numel())
    return metric[0] / metric[1]

max_degree = 20
n_train, n_test = 100, 100
true_w = np.zeros(max_degree) # 分配大量的空间 [20,1]
true_w[0:4] = np.array([5,1.2,-3.4,5.6])
# 生成一个形状为(n_test+n_train,1)的符合高斯分布的张量
features = np.random.normal(size=(n_test+n_train,1)) # [200,1]
# 打乱features
np.random.shuffle(features)
# 对每一个feature开0-200次幂不等，将np.arange转成1行n列
# poly_features 是一个200行20列的张量
# 这里的np.arange(max_degree).reshape(1,-1) 生成了 [1,20]的张量
# 这里的power利用了广播机制，对[200,1]中的每一个分别进行了1-20次的幂操作
# 最终形成了 [200,20] 的 张量 poly_features
poly_features = np.power(features,np.arange(max_degree).reshape(1, -1))

for i in range(max_degree):
    poly_features[:,i] /= math.gamma(i+1)

labels = np.dot(poly_features,true_w) # [200,20] . [20,1] -> [200,1]
labels += np.random.normal(scale=0.1,size=labels.shape)

true_w, features, poly_features, labels = [torch.tensor(x, dtype=
    torch.float32) for x in [true_w, features, poly_features, labels]]

train_loss = []
test_loss = []

def train(index,train_features, test_features, train_labels,test_labels,num_epochs=400):
    loss = nn.MSELoss(reduction = 'none') # none 意味着不进行规约，返回每个样本的损失值组成的张量
    input_shape = train_features.shape[-1]
    net = nn.Sequential(nn.Linear(input_shape,1,bias=False))
    batch_size = min(10,train_labels.shape[0])
    train_iter = d2l.load_array((train_features, train_labels.reshape((-1,1))),
                                batch_size)
    test_iter = d2l.load_array((test_features, test_labels.reshape(-1,1)),
                               batch_size, is_train=False)
    trainer = torch.optim.SGD(net.parameters(),lr = 0.01)
    train_loss.append(0)
    test_loss.append(0)
    for epoch in range(num_epochs):
        dl.train_epoch_ch3(net,train_iter=train_iter,test_iter=test_iter,loss=loss,updater=trainer)
    train_loss[index] = evaluate_loss(net,train_iter,loss)
    test_loss[index] = evaluate_loss(net,test_iter,loss)

for i in range(1,max_degree):
    train(i-1,poly_features[:n_train, :i], poly_features[n_train:, :i],
      labels[:n_train], labels[n_train:])

plt.plot([i for i in range(1,max_degree)],train_loss,label='train_loss')
plt.plot([i for i in range(1,max_degree)],test_loss,label='test_loss')
plt.xticks(list(range(1,20,2)))
plt.grid()
plt.show()

print(train_loss.index(min(train_loss)))
print(test_loss.index(min(test_loss)))