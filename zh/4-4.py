import math
import numpy as np
import torch
from torch import nn

from d2l import torch as d2l

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

def train(train_features, test_features, train_labels,test_labels,num_epochs=400):
    loss = nn.MSELoss(reduction = 'none') # none 意味着不进行规约，返回每个样本的损失值组成的张量
    input_shape = train_features.shape[-1]
    net = nn.Sequential(nn.Linear(input_shape,1,bias=False))
    batch_size = min(10,train_labels.shape[0])
    train_iter = d2l.load_array((train_features, train_labels.reshape((-1,1))),
                                batch_size)
    test_iter = d2l.load_array((test_features, test_labels.reshape(-1,1)),
                               batch_size, is_train=False)
    trainer = torch.optim.SGD(net.parameters(),lr = 0.01)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', yscale='log',
                            xlim=[1, num_epochs], ylim=[1e-3, 1e2],
                            legend=['train', 'test'])
    for epoch in range(num_epochs):
        dl.train_epoch_ch3(net,train_iter=train_iter,test_iter=test_iter,loss=loss,updater=trainer)
        if epoch==0 or (epoch+1) % 20 == 0:
            animator.add(epoch+1,(evaluate_loss(net,train_iter,loss),
                                  evaluate_loss(net,test_iter,loss)))
    print('weight:',net[0].weight.data.numpy())


# 从多项式特征中选择前4个维度，即1,x,x^2/2!,x^3/3!
train(poly_features[:n_train, :2], poly_features[n_train:, :2],
      labels[:n_train], labels[n_train:])
d2l.plt.show()