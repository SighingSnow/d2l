import torch
from torch import nn
from d2l import torch as d2l

import sys
sys.path.append('..')
from dl import torch_zh as dl

def dropout_layer(X,dropout):
    assert 0 <= dropout <= 1
    if dropout == 0: # nothing is dropped out
        return X   
    if dropout == 1:
        return torch.zeros_like(X) # everything is dropped out
    mask = (torch.rand(X.shape) > dropout).float()
    # 保证输出的期望值，因为原有的部分数值被mask掉了
    # 通过 除以(1.0-dropout) 可以恢复被mask的数值
    return mask * X / (1.0-dropout) 

dropout1,dropout2=0.5,0.2

class Net(nn.Module):
    def __init__(self,num_inputs,num_outputs,num_hiddens1,num_hiddens2,is_training=True):
        super(Net,self).__init__()
        self.num_inputs = num_inputs
        self.training = is_training
        self.lin1 = nn.Linear(num_inputs,num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1,num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2,num_outputs)
        self.relu = nn.ReLU()
    
    def forward(self,X):
        H1 = self.relu(self.lin1(X.reshape((-1,self.num_inputs))))
        if self.training:
            H1 = dropout_layer(H1,dropout1)
        H2 = self.relu(self.lin2(H1))
        if self.training:
            H2 = dropout_layer(H2,dropout2)
        out = self.lin3(H2)
        return out

num_inputs,num_outputs,num_hiddens1,num_hiddens2 = 784,10,256,256
net = Net(num_inputs,num_outputs,num_hiddens1,num_hiddens2)
num_epochs,lr,batch_size = 10,0.5,256
loss = nn.CrossEntropyLoss(reduction='none')
train_iter,test_iter = dl.load_data_fashion_mnist(batch_size)
trainer = torch.optim.SGD(net.parameters(),lr=lr)
dl.train_ch3(net,train_iter,test_iter,loss,num_epochs,trainer)
d2l.plt.show()