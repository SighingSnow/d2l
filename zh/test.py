import torch
y = torch.tensor([0,2])
y_hat = torch.tensor(([0.1,0.3,0.6],[0.3,0.2,0.5]))

if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
    y_hat = y_hat.argmax(axis=1)
    print(y_hat)
print(y_hat.dtype,y.dtype)
cmp = y_hat.type(y.dtype) == y
print(y_hat.dtype,cmp.dtype)
print(cmp)
print(cmp.type(y.dtype).sum())
print(cmp.dtype)
print(float(cmp.type(y.dtype).sum())) 