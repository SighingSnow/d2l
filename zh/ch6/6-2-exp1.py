import torch
from torch import nn
from d2l import torch as d2l

def corr2d(X,K):
    h,w = K.shape
    Y = torch.zeros((X.shape[0]-h+1,X.shape[1]-w+1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i,j] = (X[i:i+h,j:j+w] * K).sum()
    return Y

X = torch.eye(6) # a diagnal matrix
K = torch.tensor([[1.0,-1.0]])
Y = corr2d(X,K)
# print(Y)
Y = corr2d(X.T,K)
Y = corr2d(X,K.T)
print(Y)