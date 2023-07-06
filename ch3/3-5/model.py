import torch
from d2l import torch as d2l

# torch version
class LinearRegression(d2l.Module):
    def __init__(self,lr):
        super().__init__()
        self.save_hyperparameters()
        # 注意和Linear区分
        # LazyLinear需要
        self.net = torch.nn.LazyLinear(1)
        self.net.weight.data.normal_(0,0.01)
        self.net.bias.data.fill_(0)
    
    def forward(self,X):
        return self.net(X)
    
    def loss(self,y_hat,y):
        fn = torch.nn.MSELoss() 
        return fn(y_hat,y) # L2 distance，note it't loss.mean() not loss.sum()
    
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(),self.lr)
    
    def get_w_b(self):
        return (self.net.weight.data,self.net.bias.data)
    
    def get_w_grad(self):
        return self.net.weight.grad