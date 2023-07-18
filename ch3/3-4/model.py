import torch
from opt import *
from d2l import torch as d2l

class LinearRegressionScratch(d2l.Module):
    def __init__(self,num_inputs,lr,sigma=0.01):
        super().__init__()
        # 保存
        self.save_hyperparameters()
        self.w = torch.normal(0,sigma,(num_inputs,1),requires_grad=True)
        self.b = torch.zeros(1,requires_grad=True)

    def forward(self,X):
        # y = Wx + b
        return torch.matmul(X,self.w) + self.b
    
    # def loss(self,y_hat,y):
    #     l =  (y_hat - y ) ** 2 / 2
    #     # 返回的是张量的均值
    #     return l.mean()
    
    def loss(self,y_hat,y):
        l = (y_hat - y).abs().sum() / 2
        return l.mean()

    def configure_optimizers(self):
        return SGD([self.w,self.b],self.lr)