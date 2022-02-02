import torch
import os

class LazyDimension():
    def __init__(self, dim):
        self._dim = dim
        # TODO: construct an IR node
        self_node = None

    def __mul__(self, other):
        raise RuntimeError("Not Yet Implemented")

    def __div__(self, other):
        raise RuntimeError("Not Yet Implemented")

    def __int__(self):
        raise RuntimeError("Not Yet Implemented")

    def __str__(self) -> str:
        return f"LazyDimension({self._dim})"

    def __repr__(self):
        return str(self)

class LazyTensor(torch.Tensor):

    def size(self, *args):
        return [LazyDimension(dim) for dim in range(len(super().size()))]

if os.environ["LTC_ENABLE_DYNAMIC_SHAPES"]:
    torch._C._register_py_class_for_device("lazy", LazyTensor)


