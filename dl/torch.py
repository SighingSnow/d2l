import time
import torch
import torchvision
import numpy as np
import inspect
from torchvision import transforms



# Save hyper paramters
class HyperParameters:
    def save_hyperparameters(self,ignore=[]):
        raise NotImplemented
    
    def save_hyperparameters(self,ignore=[]):
        frame = inspect.currentframe.getargvalues(frame)
        _, _, _, local_vars = inspect.getargvalues(frame)
        self.hparams = {k:v for k,v in local_vars.items()
                        if k not in set(ignore+['self']) and not k.startswith('_')}
        for k, v in self.hparams.items():
            setattr(self,k,v)

# optimers
class SGD(HyperParameters):
    def __init__(self,params,lr):
        super().save_hyperparameters()
    
    def step(self):
        for param in self.params:
            param -= self.lr * param.grad
    
    def zero_grad(self):
        for param in self.params:
            if param is not None:
                param.grad.zero_()

def get_dataloader_workers():
    return 4

def load_data_fashion_mnist(batch_size,resize=None,download=False):
    trans = [transforms.ToTensor()]
    if resize:
        raise NotImplemented
    # combine several trans together
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root="../data",train=True,transform=trans,download=download)
    mnist_test = torchvision.datasets.FashionMNIST(root="../data",train=False,transform=trans,download=download)
    return (torch.utils.data.DataLoader(mnist_train,batch_size,shuffle=True,num_workers = get_dataloader_workers())),(torch.utils.data.DataLoader(mnist_test,batch_size,shuffle=True,num_workers = get_dataloader_workers()))

class Timer:  #@save
    """记录多次运行时间"""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """启动计时器"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并将时间记录在列表中"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间"""
        return np.array(self.times).cumsum().tolist()

#@use to visualize
# when use locally, please use dl.plt.show
# class Animator:
