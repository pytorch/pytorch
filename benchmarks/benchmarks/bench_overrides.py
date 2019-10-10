from .common import Benchmark

import torch
from torch import Tensor

class DuckTensor(object):
    def __torch_function__(self, func, args=None, kwargs=None):
        pass

HANDLED_FUNCTIONS = {}

def implements(torch_function):
    "Register an implementation of a torch function for a Tensor-like object."
    def decorator(func):
        HANDLED_FUNCTIONS[torch_function.__name__] = func
        return func
    return decorator

class SubTensor(Tensor):
    def __torch_function__(self, func, args=None, kwargs=None):
        if(kwargs is None):
            kwargs = {}

        if func not in HANDLED_FUNCTIONS:
            return NotImplemented
        # Note: this allows subclasses that don't override
        # __torch_function__ to handle DiagonalTensor objects.
        return HANDLED_FUNCTIONS[func](*args, **kwargs)

@implements(torch.add)
def add(mat1, mat2):
    "Implementation of torch.mm for DiagonalTensor objects"
    return 0

@implements(torch.mm)
def mm(mat1, mat2):
    "Implementation of torch.mm for DiagonalTensor objects"
    return 1

class TorchFunction(Benchmark):

    def setup(self):
        self.t1 = torch.ones(2, 2, dtype=torch.float32)
        self.t2 = torch.zeros(2, 2, dtype=torch.float32)
        self.t3 = SubTensor([[1, 1], [1, 1.]])
        self.t4 = SubTensor([[0, 0], [0, 0.]])

    def time_add(self):
        torch.add(self.t1, self.t2)

    def time_matmul(self):
        torch.mm(self.t1, self.t2)

    def time_subtensor_add(self):
        torch.add(self.t3, self.t4)

    def time_subtensor_multipy(self):
        torch.mm(self.t3, self.t4)
