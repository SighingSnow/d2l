import d2l
from d2l import torch as d2l
import numpy as np

board = d2l.ProgressBoard('x')
for x in np.arange(0,10,0.1):
    board.draw(x,np.sin(x),'sin',every_n=2)
    board.draw(x,np.cos(x),'cos',every_n=10)

d2l.plot.show()
