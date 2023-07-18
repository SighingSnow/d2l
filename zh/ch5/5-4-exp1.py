import torch
from torch import nn

class DownScaleLayer(nn.Module):
    def __init__(self,k,x_dim) -> None:
        super().__init__()
        # 当使用nn.Parameter时，该参数会自动存入
        self.k = k
        self.weight = nn.Parameter(torch.randn((k,x_dim,x_dim)))
    
    def forward(self,X):
        y = torch.zeros(self.k)
        X1 = X.sum(axis=1).unsqueeze(1)
        X2 = X.sum(axis=0).unsqueeze(1).T
        print(X1)
        print(X2)
        matrix = X.sum(axis=1).unsqueeze(1) @ X.sum(axis=0).unsqueeze(1).T
        for i in range(self.k):
            y[i] = (matrix * self.weight[i]).sum()
        return y

X = torch.randn(2)
net = DownScaleLayer(1,2)
print(net(X))