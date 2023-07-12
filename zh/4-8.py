import torch
from torch import nn
from matplotlib import pyplot as plt

x = torch.arange(-8.0,8.0,0.1,requires_grad=True)
y = torch.sigmoid(x)
y.backward(torch.ones_like(x))

# plt.plot(x.detach().numpy(),[y.detach().numpy(),x.grad.numpy()],label = ['y','grad'])
plt.plot(x.detach().numpy(),y.detach().numpy(),label = ['y'])
plt.plot(x.detach().numpy(),x.grad.numpy(),label = ['grad'])
plt.xlim((-8.0,8.0))
plt.ylim((-0.1,1.1))
plt.grid()
plt.show()