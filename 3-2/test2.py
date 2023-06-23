from hyper_parameters import *
from d2l import torch as d2l
class B(d2l.HyperParameters):
    def __init__(self,a,b,c):
        self.save_hyperparameters(ignore=['c'])
        print("self.a =",self.a,'self.b =',self.b)
        print("There is no self.c = ", not hasattr(self,'c'))

b = B(a=1,b=2,c=3)
