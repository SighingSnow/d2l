import torch
from torch import nn
from d2l import torch as d2l

T = 1000
time = torch.arange(1,T+1,dtype=torch.float32)
x = torch.sin(0.01*time) + torch.normal(0,0.2,(T,))
# d2l.plot(time,[x],'x',xlim=[1,1000],figsize=(6,3))
# d2l.plt.show()

# 也就是x[4]需要x[0:3],所以此时x[4]是label，x[0:3]是features
tau = 32
# features.shape = (996,4)
# 嵌入维度tau=4 -> 我们缺少了tau个样本
features = torch.zeros((T-tau,tau)) 
for i in range(tau): 
    # features[:,0] = x[0:996]
    # features[:,1] = x[1:997]
    # features[:,2] = x[2:998]
    # features[:,3] = x[3:999] 
    features[:,i] = x[i:T-tau+i] # x[i:T-tau+1]是features[:,i]
# labels.shape = (996,1
# ‘=)
# labels = x[4:1000]
labels = x[tau:].reshape((-1,1))

batch_size, n_train = 16,600
train_iter = d2l.load_array((features[:n_train],labels[:n_train]),batch_size,True)

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

def get_net():
    net = nn.Sequential(nn.Linear(tau,10),
                        nn.ReLU(),
                        nn.Linear(10,1))
    net.apply(init_weights)
    return net

loss = nn.MSELoss(reduction='none')

def train(net,train_iter,loss,epochs,lr):
    trainer = torch.optim.Adam(net.parameters(),lr)
    for epoch in range(epochs):
        for X,y in train_iter:
            trainer.zero_grad()
            l = loss(net(X),y)
            l.sum().backward()
            trainer.step()
        print(f'epoch {epoch+1}, '
              f'loss: {d2l.evaluate_loss(net,train_iter,loss):f}')

net = get_net()
train(net,train_iter,loss,5,0.01)

# 单步预测
onestep_preds = net(features)

# 多步预测，即利用上一步的预测结果预测下一步，
# 比如我们利用 x601,x602,x603以及x604预测了x605
# 我们需要利用预测的x605以及x602、x603、x604进行预测
multistep_preds = torch.zeros(T)
multistep_preds[:n_train+tau] = x[:n_train+tau]
for i in range(n_train+tau,T):
    multistep_preds[i]=net(multistep_preds[i-tau:i].reshape((1,-1)))

d2l.plot([time,time[tau:],time[n_train+tau:]],
        [x.detach().numpy(),onestep_preds.detach().numpy(),
         multistep_preds[n_train+tau:].detach().numpy()],'time','x',
         legend=['data','1-step','multistep'],xlim=[1,1000],
         figsize=(6,3))
d2l.plt.show()