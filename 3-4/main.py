from model import *
from dataloader import *
from trainer import *
from d2l import torch as d2l
model = LinearRegressionScratch(2,lr=0.05)
data = d2l.SyntheticRegressionData(w=torch.tensor([2,-3.4]),b = 4.2)
trainer = d2l.Trainer(max_epochs=3)
trainer.fit(model,data)

d2l.plt.show()

# 这里和真实值的差距是要比中文版大的
# 但是讲义里的意思是并不是要恢复为真实值，而是在预测的时候实现准确。
print(f'error in estimating w: {data.w - model.w.reshape(data.w.shape)}')
print(f'error in estimating b: {data.b - model.b}')