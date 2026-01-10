# from . import mock_module3
import torch
from . import mock_module3


class Class1:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def method2(self, x):
        return mock_module3.method1([], x)


def method1(x, y):
    torch.ones(1, 1)
    x.append(y)
    return x
