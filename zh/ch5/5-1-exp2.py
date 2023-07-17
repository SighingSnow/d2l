import torch
from torch import nn


class n3(nn.Module):
    def __init__(self,b1,b2):
        super(n3,self).__init__()
        self.b1 = b1
        self.b2 = b2
    
    def forward(self,X):
        X = torch.cat((self.b1(X),self.b2(X)))
        return X

b = n3(nn.Linear(20,5),nn.Linear(20,10))
net = nn.Sequential(b,nn.ReLU(),nn.Linear(15,5))
X = torch.rand(20)
print(net(X).shape)