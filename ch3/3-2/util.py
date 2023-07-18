import time
import numpy as np
import torch
from torch import nn
from d2l import torch as d2l

def add_to_class(Class): 
    def wrapper(obj):
        setattr(Class,obj.__name__,obj)
    return wrapper
