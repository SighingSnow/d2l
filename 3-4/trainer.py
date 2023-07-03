import torch
from d2l import torch as d2l

class Trainer(d2l.HyperParameters):
    def __init__(self,max_epoches,num_gpus=0,gradient_clip_val=0):
        self.save_hyperparameters()
        assert num_gpus==0, "No GPU support yet"

    def prepare_data(self,data):
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = (len(self.val_dataloader) if self.val_dataloader is not None else 0)

    def prepare_model(self,model):
        model.trainer = self
        model.board.xlim = [0,self.max_epoches]
        self.model = model

    # 与训练集相适应
    def fit(self,model,data):
        self.prepare_data(data) # 准备数据
        self.prepare_model(model) # 准备模型
        self.optim = model.configure_optimizers() # 选择合适的梯度下降算法
        self.epoch = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        for self.epoch in range(self.max_epoches):
            self.fit_epoch()

    def prepare_batch(self,batch):
        return batch

    def fit_epoch(self):
        self.model.train() # how to train
        for batch in self.train_dataloader:
            loss = self.model.training_step(self.prepare_batch(batch))
            self.optim.zero_grad()
            # 上下文管理器，确保在推理过程中不会进行梯度计算
            with torch.no_grad():
                loss.backward()
                if self.gradient_clip_val > 0:
                    self.clip_gradients(self.gradient_clip_val,self.model)
                self.optim.step()
            self.train_batch_idx += 1
        if self.val_dataloader is None :
            return 
        self.model.eval()
        for batch in self.val_dataloader:
            with torch.no_grad():
                self.model.validation_step(self.prepare_batch)
            self.val_batch_idx += 1