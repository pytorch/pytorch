code = """
import torch
from torch.autograd import Variable, Function

x = torch.ones(5, 5, requires_grad=True)

class MyOp(Function):
    @staticmethod
    def forward(self, x):
        return x + 1

    @staticmethod
    def backward(self, dy):
        return dy
"""

import timeit
REP = 20
print(timeit.repeat("""
y = x
lst = []
for i in range(10000):
  y = MyOp.apply(y)
  lst.append(y)
""", setup=code, number=3, repeat=REP)[REP/2])

