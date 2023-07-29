import torch 
from torch import nn
from d2l import torch as d2l

import sys
sys.path.append('..')
from dl import torch_zh as dl

from matplotlib import pyplot as plt

def l2_penalty(w):
    return (w**2).sum() / 2;

class Data(d2l.DataModule):
    def __init__(self,num_train,num_val,num_inputs,batch_size):
        self.save_hyperparameters()
        n = num_train + num_val
        self.X = torch.randn(n,num_inputs)
        noise = torch.randn(n,1) * 0.01 # noise
        w,b = torch.ones((num_inputs,1)) * 0.01, 0.05
        self.y = self.X@w + b + noise
    
    def get_dataloader(self, train):
        i = slice(0,self.num_train) if train else slice(self.num_train,None)
        return self.get_tensorloader([self.X,self.y],train,i)

class WeighDecayScratch(dl.LinearRegressScratch):
    def __init__(self,num_inputs,lambd,lr,sigma=0.01):
        super().__init__(num_inputs,lr,sigma)
        self.save_hyperparameters()

    def loss(self,y_hat,y):
        penaty = l2_penalty(self.w)
        return super().loss(y_hat,y) + self.lambd * penaty

data = Data(num_train=20,num_val=100,num_inputs=200,batch_size=5)
trainer = dl.Trainer(max_epochs=10)

train_acc_lst = []
val_acc_lst = []

def accuracy(y_hat,y):
    return (1 - ((y_hat-y).mean()/y.mean().abs())) * 100

def trainer_scratch(index,lambd):
    model = WeighDecayScratch(num_inputs=200,lambd=lambd,lr=0.01)
    model.board.yscale = 'log'
    trainer.fit(model,data)
    # print('L2 norm of W:',float(l2_penalty(model.w)))
    y_hat = model.forward(data.X)
    # train_acc_lst[index] = accuracy(y_hat[:data.num_train],data.y[:data.num_train])
    # val_acc_lst[index] = accuracy(y_hat[data.num_train],data.y[data.num_train])
    return accuracy(y_hat[:data.num_train],data.y[:data.num_train]), accuracy(y_hat[data.num_train],data.y[data.num_train])

for i in range(11):
    x,y = trainer_scratch(i,i)
    train_acc_lst.append(x)
    val_acc_lst.append(y)

train_acc_lst = [ x.detach().numpy() for x in train_acc_lst]
val_acc_lst = [ x.detach().numpy() for x in val_acc_lst]

plt.plot([i for i in range(11)],train_acc_lst,label='train accuracy')
plt.plot([i for i in range(11)],val_acc_lst,label='test accuracy')
plt.xticks(list(range(0,11)))
plt.grid()
plt.show()