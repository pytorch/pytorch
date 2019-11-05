from .common import Benchmark

import torch
from torch import Tensor

class SubTensorWithoutTorchFunction(Tensor):
    pass

HANDLED_FUNCTIONS = {}

def implements(torch_function):
    "Register an implementation of a torch function for a Tensor-like object."
    def decorator(func):
        HANDLED_FUNCTIONS[torch_function] = func
        return func
    return decorator

class SubTensorWithTorchFunction(Tensor):
    def __torch_function__(self, func, args=(), kwargs=None):
        if(kwargs is None):
            kwargs = {}

        if func not in HANDLED_FUNCTIONS:
            return NotImplemented

        return HANDLED_FUNCTIONS[func](*args, **kwargs)

# define these at module scope since we want them to be available in
# SubTensorWithTorchFunction's implementation of add and mm
t1 = torch.tensor([[1, 1], [1, 1.]])
t2 = torch.tensor([[0, 0], [0, 0.]])

# define add and mm to use global versions of t1 and t2 so that we do an
# apples-to-apples comparison with the pure-tensor case. This means any excess
# time in the benchmarks using SubTensorWithTorchFunction is overhead for
# calling the __torch_function__ implementation
@implements(torch.add)
def add(mat1, mat2):
    return torch.add(t1, t2)

@implements(torch.norm)
def norm(mat1):
    return torch.norm(t1)

@implements(torch.abs)
def abs(mat1):
    return torch.abs(t1)

class TorchFunction(Benchmark):

    def setup(self):
        self.t1 = torch.tensor([[1, 1], [1, 1.]])
        self.t2 = torch.tensor([[0, 0], [0, 0.]])
        self.t3 = SubTensorWithTorchFunction([[1, 1], [1, 1.]])
        self.t4 = SubTensorWithTorchFunction([[0, 0], [0, 0.]])
        self.t5 = SubTensorWithoutTorchFunction([[1, 1], [1, 1.]])
        self.t6 = SubTensorWithoutTorchFunction([[0, 0], [0, 0.]])

    def time_tensor_tensor_add(self):
        torch.add(self.t1, self.t2)

    def time_tensor_norm(self):
        torch.norm(self.t1)

    def time_tensor_abs(self):
        torch.abs(self.t1)

    def time_withtf_withtf_add(self):
        torch.add(self.t3, self.t4)

    def time_withtf_tensor_add(self):
        torch.add(self.t3, self.t1)

    def time_tensor_withtf_add(self):
        torch.add(self.t1, self.t3)

    def time_withtf_norm(self):
        torch.norm(self.t3)

    def time_withtf_abs(self):
        torch.abs(self.t3)

    def time_withouttf_withouttf_add(self):
        torch.add(self.t5, self.t6)

    def time_withouttf_withtf_add(self):
        torch.add(self.t5, self.t4)

    def time_withtf_withouttf_add(self):
        torch.add(self.t4, self.t5)

    def time_withouttf_tensor_add(self):
        torch.add(self.t5, self.t2)

    def time_tensor_withouttf_add(self):
        torch.add(self.t2, self.t5)

    def time_withouttf_norm(self):
        torch.norm(self.t5)

    def time_without_abs(self):
        torch.abs(self.t5)
