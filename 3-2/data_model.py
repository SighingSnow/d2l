from d2l import torch as d2l

class DataModel(d2l.HyperParameters):
    def __init__(self,root="../data",num_workers=1):
        self.save_hyperparameters()

    def get_dataloader(self,train):
        raise NotImplementedError

    def train_dataloader(self):
        return self.get_dataloader(train=True)

    def val_dataloader(self):
        return self.get_dataloader(train=False)
