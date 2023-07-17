import torch
from torch import nn

net = nn.Sequential(nn.Linear(4,8),nn.ReLU(),nn.Linear(8,1))
X = torch.rand(size=(2,4))
net(X)

# print(net[2].state_dict()) # 第二个全连接层的参数

# print(type(net[2].bias))
# print(net[2].bias)
# print(net[2].bias.data)

# 一次性访问所有参数
# print(*[(name,param.shape) for name,param in net[0].named_parameters()])
# print(*[(name,param.shape) for name,param in net.named_parameters()])

def block1():
    return nn.Sequential(nn.Linear(4,8),nn.ReLU(),
                         nn.Linear(8,4),nn.ReLU())

def block2():
    net = nn.Sequential()
    for i in range(4):
        net.add_module(f'block {i}',block1())
    return net

rgnet = nn.Sequential(block2(),nn.Linear(4,1))
# print(rgnet)

# 参数初始化
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.zeros_(m.bias)
        nn.init.normal_(m.weight,mean=0,std=0.01)
net.apply(init_normal)
print(net[0].weight.data,net[0].bias.data)