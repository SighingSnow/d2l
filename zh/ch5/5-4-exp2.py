import torch
from torch import nn
# 这份代码是抄的
class HalfFFT(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,X):
        half_len = round(X.shape[1] / 2)
        X_f = torch.fft.fft(X) # Flourier
        return X_f[:,:half_len]

net = HalfFFT()
print(net(torch.rand(2,3)))