from __future__ import absolute_import, division, print_function

from .common import Benchmark

from torch import Tensor
from torch._overrides import torch_function_dispatch


def _broadcast_tensors_dispatcher(*tensors):
    return (Tensor,)


@torch_function_dispatch(_broadcast_tensors_dispatcher)
def mock_broadcast_tensors(*tensors):
    pass


def _concatenate_dispatcher(tensors, dim=None, out=None):
    for tensor in tensors:
        yield tensor
    if out is not None:
        yield out


@torch_function_dispatch(_concatenate_dispatcher)
def mock_concatenate(tensors, dim=0, out=None):
    pass


class DuckTensor(object):
    def __torch_function__(self, func, types, args, kwargs):
        pass


class TorchFunction(Benchmark):

    def setup(self):
        self.torch_tensor = Tensor(1)
        self.torch_tensors = [Tensor(1), Tensor(2)]
        self.many_tensors = 500 * self.torch_tensors
        self.duck_tensor = DuckTensor()
        self.duck_tensors = [DuckTensor(), DuckTensor()]
        self.mixed_tensors = [Tensor(1), DuckTensor()]

    def time_mock_broadcast_tensors_torch(self):
        mock_broadcast_tensors(self.torch_tensor, ())

    def time_mock_broadcast_tensors_duck(self):
        mock_broadcast_tensors(self.duck_tensor, ())

    def time_mock_concatenate_torch(self):
        mock_concatenate(self.torch_tensors, dim=0)

    def time_mock_concatenate_many(self):
        mock_concatenate(self.many_tensors, dim=0)

    def time_mock_concatenate_duck(self):
        mock_concatenate(self.duck_tensors, dim=0)

    def time_mock_concatenate_mixed(self):
        mock_concatenate(self.mixed_tensors, dim=0)
