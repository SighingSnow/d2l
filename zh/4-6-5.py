import torch
from torch import nn
from d2l import torch as d2l
import sys
sys.path.append('..')
from dl import torch_zh as dl

num_inputs,num_outputs,num_hiddens1,num_hiddens2 = 784,10,256,256
dropout2,dropout1 = 0.5,0.2
net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(num_inputs,num_hiddens1),
    nn.ReLU(),
    nn.Dropout(dropout1),
    nn.Linear(num_hiddens1,num_hiddens2),
    nn.ReLU(),
    nn.Dropout(dropout2),
    nn.Linear(num_hiddens2,num_outputs)
)

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight,std=0.01)

net.apply(init_weights)
num_epochs,lr,batch_size = 20, 0.5,256
loss = nn.CrossEntropyLoss(reduction='none')
train_iter,test_iter = dl.load_data_fashion_mnist(batch_size)
trainer = torch.optim.SGD(net.parameters(),lr=lr,weight_decay=0.005)
dl.train_ch3(net,train_iter,test_iter,loss,num_epochs,trainer)
d2l.plt.show()