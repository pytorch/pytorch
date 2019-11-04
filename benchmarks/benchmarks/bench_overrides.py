from .common import Benchmark

import torch
from torch import Tensor

class DuckTensor(object):
    def __torch_function__(self, func, args=(), kwargs=None):
        pass

HANDLED_FUNCTIONS = {}

def implements(torch_function):
    "Register an implementation of a torch function for a Tensor-like object."
    def decorator(func):
        HANDLED_FUNCTIONS[torch_function] = func
        return func
    return decorator

class SubTensor(Tensor):
    def __torch_function__(self, func, args=(), kwargs=None):
        if(kwargs is None):
            kwargs = {}

        if func not in HANDLED_FUNCTIONS:
            return NotImplemented

        return HANDLED_FUNCTIONS[func](*args, **kwargs)

# define these at module scope since we want them to be
# available in SubTensor's implementation of add and mm
t1 = torch.tensor([[1, 1], [1, 1.]])
t2 = torch.tensor([[0, 0], [0, 0.]])

# define add and mm to use global versions of t1 and t2 so that we do an
# apples-to-apples comparison with the pure-tensor case. This means any
# excess time in the benchmarks using SubTensor is overhead for calling
# the __torch_function__ implementation
@implements(torch.add)
def add(mat1, mat2):
    return torch.add(t1, t2)

@implements(torch.mm)
def mm(mat1, mat2):
    return torch.mm(t1, t2)

class TorchFunction(Benchmark):

    def setup(self):
        self.t1 = torch.tensor([[1, 1], [1, 1.]])
        self.t2 = torch.tensor([[0, 0], [0, 0.]])
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
