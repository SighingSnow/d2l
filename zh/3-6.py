import torch
from IPython import display
from d2l import torch as d2l

import sys
sys.path.append("..")
from dl import torch as dl # my own dl package

batch_size = 256
# 加载
train_iter,test_iter = dl.load_data_fashion_mnist(batch_size)

num_inputs = 28 * 28 # fig_size = 28 * 28
num_outputs = 10

W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs,requires_grad=True)
print(W.grad)
class Accumulator:
    def __init__(self,n):
        self.data = [0.0] * n
    
    def add(self,*args):
        # names ['Alice','Bob]
        # ages  [24,35]
        # for item in zip(name,ages):
        #    print(item)
        # will get ('Alice',24) ('Bob',35)
        self.data = [a+float(b) for a,b in zip(self.data,args)]
    
    def reset(self):
        self.data = [0.0] * len(self.data)
    
    def __getitem__(self,idx):
        return self.data[idx]

def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1,keepdim=True)
    return X_exp / partition

def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)

def cross_entropy(y,y_hat):
    return -torch.log(y_hat[range(len(y_hat)),y])

def accuracy(y_hat,y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        # 选出最大的
        y_hat = y_hat.argmax(axis=1)
    # 需要将y_hat的类型转为y进行比较，否则可能会有2.0!=0的可能
    cmp = y_hat.type(y_hat.dtype) == y
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(net,data_iter):
    # isinstance:
    # 判断net是否是torch.nn.Module
    if isinstance(net,torch.nn.Module):
        # 将模型转为评估模式
        # 在此模式下取消归一化以及dropout操作
        net.eval() 
    metric = Accumulator(2)
    with torch.no_grad():
        for X,y in data_iter:
            metric.add(accuracy(net(X),y),y.numel())
    return metric[0] / metric[1]

def train_epoch(net,train_iter,loss,updater):
    if isinstance(net,torch.nn.Module):
        net.train()
    # 训练损失总和，准确度总和，样本数
    metric = Accumulator(3)
    for X,y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y )
        if isinstance(updater,torch.optim.Optimizer):
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()),accuracy(y_hat,y),y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]

def updater(batch_size,lr):
    return d2l.sgd([W,b],lr,batch_size)

def train(net,train_iter,test_iter,loss,num_epochs,updater):
    animator = d2l.Animator(xlabel='epoch',xlim=[1,num_epochs],ylim=[0.3,0.9],legend=['train_loss','train acc','test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch(net,train_iter,loss,updater)
        test_acc = evaluate_accuracy(net,test_iter)
        animator.add((epoch+1,train_metrics+(test_acc,)))
    train_loss, train_acc = train_metrics


num_epochs = 10
lr = 0.01
update = updater(batch_size,lr)
train(net,train_iter,test_iter,cross_entropy,num_epochs,d2l.sgd([W,b],lr,batch_size))
d2l.plt.show()