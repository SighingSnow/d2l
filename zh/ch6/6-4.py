import torch
from torch import nn 

def corrd2d(X,K):
    h,w = X.shape
    kh,kw = K.shape
    Y = torch.zeros((h-kh+1,w-kw+1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i,j] = (X[i:i+kh,j:j+kw] * K).sum()
    return Y

def corr2d_multi_in(X,K):
    return sum(corrd2d(x,k) for x,k in zip(X,K))

def corr2d_multi_in_and_out(X,K):
    return torch.stack([corr2d_multi_in(X,k) for k in K],0)

X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
               [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])

print(corr2d_multi_in(X, K))

K = torch.stack((K,K+1,K+2),0)
print(corr2d_multi_in_and_out(X,K).shape)