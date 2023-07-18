import torch
from torch import nn
from torch.nn import functional as F

# 最简单的操作
x = torch.arange(4)
torch.save(x,'x-file')

x2 = torch.load('x-file')

# 保存列表并读出
y = torch.zeros(4)
torch.save([x,y],'y-files')
z1,z2 = torch.load('y-files')
print(z1,z2)

# 保存模型参数并读出
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784,256),
            nn.ReLU(),
            nn.Linear(256,10)
        )

    def forward(self,X):
        return self.net(X)

net = MLP()
X = torch.randn((1,784))
Y = net(X)
torch.save(net.state_dict(),'mlp.params')

clone = MLP()
clone.load_state_dict(torch.load('mlp.params'))
print(clone.eval())
# 可以看到通过这种方式保存的模型得到的输出是一样的
Y_clone = clone(X)
print(Y_clone == Y)