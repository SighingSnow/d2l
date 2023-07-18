from synthetic_regression_data import *
import random

# @d2l.add_to_class(SyntheticRegressionData)
# def get_dataloader(self,train):
#     if train:
#         indices = list(range(0,self.num_train))
#         random.shuffle(indices)
#     else :
#         indices = list(range(self.num_train,self.num_train+self.num_val))
#     for i in range(0,len(indices),self.batch_size):
#         batch_indices = torch.tensor(indices[i:i+self.batch_size])
#         yield self.X[batch_indices], self.y[batch_indices]

@d2l.add_to_class(d2l.DataModule)
def get_tensorloader(self,tensors,train,indices=slice(0,None)):
    tensors = tuple(a[indices] for a in tensors)
    dataset = torch.utils.data.TensorDataset(*tensors)
    return torch.utils.data.DataLoader(dataset,self.batch_size,shuffle=train)

@d2l.add_to_class(SyntheticRegressionData)
def get_dataloader(self,train):
    i = slice(0,self.num_train) if train else slice(self.num_train,self.num_train+self.num_val)
    return self.get_tensorloader((self.X,self.y),train,i)

@d2l.add_to_class(SyntheticRegressionData)
def get_dataloader_onfly(self,train):
    if train:
        X = torch.randn(range(0,self.num_train))
    else:
        X = torch.randn(range(self.num_train,self.num_val+self.num_train))
    return X,y

data = SyntheticRegressionData(w=torch.tensor([2,-3.4]),b=4.2)

X,y = next(iter(data.train_dataloader()))
print('X shape:',X.shape,'\ny shape:',y.shape)