import torch 
import torchvision

from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l

import sys
sys.path.append("..")
from dl.timer import *

trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST("../data",train=True,transform=trans,download=False)
mnist_test = torchvision.datasets.FashionMNIST("../data",train=False,transform=trans,download=False)

# 训练集长度和测试集长度
# print(len(mnist_train))
# print(len(mnist_test))

# shape
# mnist_train[0][0].shape

# 
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  #@save
    """绘制图像列表"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 图片张量
            ax.imshow(img.numpy())
        else:
            # PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes

# X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
# show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y));
# d2l.plt.show()

# 读取数据
def get_dataloader_workers():
    return 4
batch_size = 256
# 如果不写main函数的话会发生一种情况即
# 当前进程还没有完成引导的情况下，新进程就创建了
# 通过写main可以确保主模块在启动子进程时已经完成了初始化
# 原因是没有使用fork，可是mac上为什么会出现这个问题呢
if __name__ == '__main__':
    train_iter = data.DataLoader(mnist_train,batch_size,shuffle=True,num_workers=get_dataloader_workers())
    timer = Timer() # from my own dl.Timer
    for X,y in train_iter:
        continue
    print(f'{timer.stop():.2f} sec')