import torch 
from torch import nn

class MySequential(nn.Module):
    def __init__(self,*args):
        super().__init__()
        # 注意这里的enumerate，如果不加enumerate会导致args是无法枚举的情况
        for idx,model in enumerate(args):
            self._modules[str(idx)] = model

    def forward(self,X):
        for key, model in self._modules.items():
            X = model(X)
        return X

class MLP(nn.Module):
    def __init__(self):
        super().__init__();
        self.hidden = nn.Linear(20,256)
        self.out = nn.Linear(256,10)
    
    def forward(self,X):
        return self.out(nn.functional.relu(self.hidden(X)))

class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.rand((20,20),requires_grad=False)
        self.linear = nn.Linear(20,20)
    
    def forward(self,X):
        X = self.linear(X)
        X = nn.functional.relu(torch.mm(X,self.w)+1)
        X = self.linear(X)
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()
# exp1
class ListMLP(nn.Module):
    def __init__(self,*args):
        super().__init__()
        self.models = []
        for idx,model in enumerate(args):
            self.models.append(model)
    
    def forward(self,X):
        for model in self.models:
            X = model(X)
        return X

linear1 = nn.Linear(20,256)
relu = nn.ReLU()
linear2 = nn.Linear(256,10)
norm_mlp = MySequential(linear1,relu,linear2)
list_mlp = ListMLP(linear1,relu,linear2)
# X = torch.randn(20) 随机数符合正态分布
X = torch.rand(20) # 0-1之间的随机分布

print(norm_mlp)
print(list_mlp)

print(norm_mlp(X))
print(list_mlp(X))