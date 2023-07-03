import random
import torch
from d2l import torch as d2l

def synthetic_data(w,b,num_examples):
    x = torch.normal(0,1,(num_examples,len(w)))
    y = torch.matmul(x,w) + b
    y += torch.normal(0,0.01,y.shape)
    return x,y.reshape((-1,1))

true_w = torch.tensor([2,-3.4])
true_b = 4.2
features, labels = synthetic_data(true_w,true_b,1000)
print('features',features[0],'\nlabel:',labels[0])

# d2l.plt.scatter(features[:,1].detach().numpy(),labels.detach().numpy(),1)
# d2l.plt.show()

# 如果 num_examples % batch_size != 0 怎么办呢
def data_iter(batch_size,features,labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0,num_examples,batch_size):
        batch_indices = torch.tensor(
            indices[i:min(i+batch_size,num_examples)]
        )
        # yield 函数可以逐个生成值，而不是一次性产生所有值
        yield features[batch_indices],labels[batch_indices]

def linreg(X,w,b):
    return torch.matmul(X,w) + b

def squared_loss(y_hat,y):
    return (y_hat - y) ** 2 / 2

def sgd(params,lr,batch_size):
    # 关闭计算图操作
    # pytorch要求对leaf variable不能进行张量操作
    # 只能写成 param = param - lr * param.grad / batch_size（该方法会导致param地址变动）
    # 而通过关闭计算图的操作，使得in-place操作可以执行
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad  / batch_size
            param.grad.zero_()

batch_size = 10
cnt = 0

# w = torch.normal(0,0.01,size=(2,1),requires_grad=True)
w = torch.zeros(size=(2,1),requires_grad=True)
b = torch.zeros(1,requires_grad=True)

lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X,y in data_iter(batch_size,features,labels):
        l = loss(net(X,w,b),y)
        l.sum().backward()
        sgd([w,b],lr,batch_size)
    with torch.no_grad():
        train_l = loss(net(features,w,b),labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')