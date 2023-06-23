from d2l import torch as d2l
class ProgressBoard(d2l.HyperParameters):
    def __init__(self,xlabel=None,ylabel=None,xlim=None,
            ylim=None,xscale='linear',yscale='linear',
            ls=[],colors=['C0','C1','C2','C3'],
            fig=None,axes=None,figsize=(3.5,2.5),display=True):
        self.save_hyperparameters()

    def draw(self,x,y,label,every_n=1):
        raise NotImplemented
