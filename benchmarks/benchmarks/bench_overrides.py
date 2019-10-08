from .common import Benchmark

import torch
from torch import Tensor

class DuckTensor(object):
    def __torch_function__(self, func, args=None, kwargs={}):
        pass

class SubTensor(Tensor):
    def __torch_function__(self, func, args=None, kwargs={}):
        pass

class TorchFunction(Benchmark):

    def setup(self):
        self.t1 = torch.ones(2, 2, dtype = torch.float32)
        self.t2 = torch.zeros(2, 2, dtype = torch.float32)

    def time_add(self):
        torch.add(self.t1, self.t2)

    def time_matmul(self):
        torch.mm(self.t1, self.t2)
