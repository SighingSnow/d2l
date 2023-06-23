import sys
sys.path.append(".")
import numpy as np
import torch
from Timer import *


n = 10000
a = torch.ones([n])
b = torch.ones([n])

c = torch.zeros(n)
timer1 = Timer()
for i in range(n):
    c[i] = a[i]+b[i]
print("{:.5f}".format(timer1.stop()))

timer2 = Timer()
d = a+b
print("{:.5f}".format(timer2.stop()))
