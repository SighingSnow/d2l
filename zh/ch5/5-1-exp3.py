import torch
from torch import nn

class largeBlock(nn.Module):
    def __init__(self,size):
        super().__init__()
        self.models = []
        self.final_layer = nn.Sequential(
            # nn.Flatten() 注意这里不能加nn.Flatten()
            # nn.Flatten主要是将高维的张量转化为适应全连接层的2维张量，但在本例中只有1维张量。
            nn.Linear(size*64,256),
            nn.ReLU(),
            nn.Linear(256,10)
        )
        for i in range(size):
            self.models.append(nn.Linear(20,64))
    
    def forward(self,X):
        lst = [model(X) for model in self.models]
        print(lst[0].type)
        X = torch.cat(lst)
        return self.final_layer(X)
    
X = torch.rand(20)
net = largeBlock(4)
print(net(X))