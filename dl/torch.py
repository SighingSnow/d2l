import time
import torch
import torchvision
import numpy as np
import inspect
from torchvision import transforms
from d2l import torch as d2l
from IPython import display
from dl import torch as dl
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

# models
class Trainer(d2l.HyperParameters):
    def __init__(self,max_epochs,num_gpus=0,gradient_clip_val=0):
        self.save_hyperparameters()
        assert num_gpus == 0, 'No GPU support yet'
    
    def prepare_data(self,data):
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = (len(self.val_dataloader) if self.val_dataloader is not None else 0)
    
    def prepare_model(self,model):
        model.trainer = self
        model.board.xlim = [0,self.max_epochs]
        # if self.num_gpus:
        #     raise NotImplementedError
        self.model = model
    
    def prepare_batch(self,batch):
        # if self.gpus:
        #     raise NotImplemented
        return batch

    def fit_epoch(self):
        self.model.train() # enable dropout and batch normalization
        for batch in self.train_dataloader:
            loss = self.model.training_step(self.prepare_batch(batch))
            self.optim.zero_grad()
            with torch.no_grad():
                loss.backward()
                if self.gradient_clip_val > 0:
                    raise NotImplemented
                self.optim.step()
            self.train_batch_idx += 1
        if self.val_dataloader is None:
            return 
        self.model.eval() # disable dropout and batch normalization
        for batch in self.val_dataloader:
            with torch.no_grad():
                self.model.validation_step(self.prepare_batch(batch))
            self.val_batch_idx+=1
    
    def fit(self,model,data):
        self.prepare_data(data)
        self.prepare_model(model)
        self.optim = model.configure_optimizers()
        self.epoch = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        for self.epoch in range(self.max_epochs):
            self.fit_epoch()
    
class LinearRegressScratch(d2l.Module):
    def __init__(self,num_inputs,lr,sigma=0.01):
        super().__init__()
        self.save_hyperparameters() # equals self.num_inputs = num_inputs etc.
        self.w = torch.normal(0,sigma,(num_inputs,1),requires_grad=True)
        self.b = torch.zeros(1,requires_grad=True)

    def forward(self,X):
        return X@self.w+self.b
    
    def loss(self,y_hat,y):
        l = (y_hat - y) ** 2 / 2
        return l.mean()
    
    def configure_optimizers(self):
        return torch.optim.SGD([self.w,self.b],lr=self.lr)

# loss
def MSELoss(y_hat,y):
    return (y_hat-y) ** 2 / 2

def crossEntryLoss(y_hat,y):
    return -torch.log(y_hat[range(len(y_hat)),y])

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

def accuracy(y_hat,y): # this is for classification not for some linear calculation
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(net,data_iter):
    if isinstance(net,torch.nn.Module):
        net.eval()
    metric = Accumulator(2)
    with torch.no_grad():
        for X,y in data_iter:
            metric.add(accuracy(net(X),y),y.numel())
    return metric[0] / metric[1]

def train_epoch_ch3(net,train_iter,test_iter,loss,updater):
    if isinstance(net,torch.nn.Module):
        net.train()
    metric = Accumulator(3)
    for X,y in train_iter:
        y_hat = net(X)
        l = loss(y_hat,y)
        if isinstance(updater,torch.optim.Optimizer):
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()),accuracy(y_hat,y),y.numel())
    return metric[0]/metric[2],metric[1]/metric[2]

def train_ch3(net,train_iter,test_iter,loss,num_epochs,updater):
    animator = Animator(xlabel='epoch',xlim=[1,num_epochs],ylim=[0.3,1.0],
                        legend=['train loss','train acc','test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net,train_iter,test_iter,loss,updater)
        test_acc = evaluate_accuracy(net,test_iter)
        animator.add(epoch+1,train_metrics+(test_acc,))
        train_loss,train_acc = train_metrics
        # assert train_loss < 0.5, train_loss
        # assert train_acc <= 1 and train_acc > 0.7 , train_acc
        # assert test_acc <= 1 and test_acc > 0.7, test_acc

def predict_ch3(net,test_iter,n=6):
    for X,y in test_iter:
        break
    trues =  d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true + '\n' + pred for true , pred in zip(trues,preds)]
    d2l.show_images(
        X[0:n].reshape((n,28,28)),1,n,titles = titles[0:n]
    )

# utilities
def l2_penalty(w):
    return (w**2).sum() / 2;

class Accumulator:
    def __init__(self,n):
        self.data = [0.0]*n
    
    def add(self,*args):
        self.data = [a+float(b) for a,b in zip(self.data,args)]
    
    def __getitem__(self,idx):
        return self.data[idx] 

class Animator:  #@save
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)

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
