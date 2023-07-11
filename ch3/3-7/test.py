import torch 
from torch import nn
from d2l import torch as d2l

import sys
sys.path.append('..')
from dl import torch as dl

def l2_penalty(w):
    return (w**2).sum() / 2;

def l1_penalty(w):
    return w.sum().abs()

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

def trainer_scratch(lambd):
    model = WeighDecayScratch(num_inputs=200,lambd=lambd,lr=0.01)
    model.board.yscale = 'log'
    trainer.fit(model,data)
    print('L2 norm of W:',float(l2_penalty(model.w)))

trainer_scratch(3)
d2l.plt.show()