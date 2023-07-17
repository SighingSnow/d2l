import torch
from torch import nn

net = nn.Sequential(nn.LazyLinear(256),nn.ReLU(),nn.LazyLinear(10))
print(net[0].weight) 

X = torch.rand(2,20)
net(X)
print(net[0].weight.data)