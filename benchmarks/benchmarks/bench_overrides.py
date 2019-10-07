from .common import Benchmark

import torch
from torch import Tensor

def mock_add(t1, t2):
   torch.add(t1, t2)

def mock_matmul(t1, t2):
   torch.mm(t1, t2)

class DuckTensor(object):
    def __torch_function__(self, func, args=None, kwargs={}):
        pass

class SubTensor(Tensor):
    def __torch_function__(self, func, args=None, kwargs={}):
        pass

class TorchFunction(Benchmark):

    def setup(self):
        self.t1 = torch.rand(1000, 1000)
        self.t2 = torch.rand(1000, 1000)

    def time_mock_add(self):
        mock_add(self.t1, self.t2)

    def time_mock_matmul(self):
        mock_matmul(self.t1, self.t2)

