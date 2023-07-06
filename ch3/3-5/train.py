from model import *

max_epochs = 3


model = LinearRegression(lr=0.01)
data = d2l.SyntheticRegressionData(w=torch.tensor([-2,3.4]),b=4.2)
trainer =  d2l.Trainer(10)
trainer.fit(model,data)

W, b = model.get_w_b()
print(f'error in estimating w:{data.w - W.reshape(data.w.shape)}')
print(f'error in estimating b:{data.b - b}')
# it's none 
# print W.grad()
print(model.get_w_grad())

d2l.plt.show()