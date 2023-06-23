from d2l import torch
from progress_board import *

class Module(nn.Module,d2l.HyperParameters):
    def __init__(self,plot_train_per_epoch=2,plot_valid_per_epoch=1):
        super().__init__()
        self.save_hyperparameters()
        self.board= ProgressBoard()

    def loss(self,y_hat,y):
        raise NotImplementedError

    def forward(self,X):
        assert hasattr(self,'net'), 'Neural network is defined'
        return self.net(X)

    def plot(self,key,value,train):
        assert hasattr(self,'trainer'), 'Trainer is not inited'
        self.board.x = 'epoch'
        if train:
            x = self.trainer.train_batch_idx / \
                self.trainer.num_train_batches
            n = self.trainer.num_train_batches / \
                self.plot_train_per_epoch
        else :
            x = self.trainer.epoch+1
            n = self.trainer.num_train_batches / \
                self.plot_valid_per_epoch
        self.board.draw(x,value.to(d2l.cpu()).detach.numpy(),
                ('train_' if train else 'val_') + key,
                every_n = int(n))

    def training_step(self,batch):
        l = self.loss()
        self.plot('loss',l,train=True)
        return l

    def validation_step(self,batch):
        l = self.loss()
        self.plot('loss',l,train=False)

    def configure_optimizers(self):
        raise NotImplementedError

