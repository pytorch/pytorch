import unittest

import torch
from torch import Tensor, SymInt
from torch.autograd import Function
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.utils._pytree import tree_map
from typing import List, Union
from functools import reduce
import operator

# Python tensor subclass version of NT.
# Properties:
#  * Constrained to contiguous-only
#  * Returned for fake / meta conversion
#  * Supports size() / stride() -> [SymInt] for dynamic shapes
#  * Supports size() / stride() -> [Union[int, List[int]]] for non-dynamic shapes
class NestedTensor(Tensor):
    @staticmethod
    def __new__(cls, buffer, *args, **kwargs):
        # Put in obvious garbage; we don't ever want to actually hit the C++ side of this thing
        return Tensor._make_subclass(cls, torch.empty(0))

    # sizes should be a list of SymInts for the dynamic shapes case
    def __init__(self, buffer: Tensor, sizes: List[Union[int, List[int], SymInt]]):
        self.buffer = buffer
        self.sizes = sizes
        self.n_tensors = self.sizes[0]
        self.total_dim = len(self.sizes)

    @classmethod
    def _sizes_from_shape_list(self, shape_list: Union[Tensor, List[torch.Size]]):
        # Produces a O(dim) size description from either a Tensor or list of shapes
        if isinstance(shape_list, Tensor):
            # Convert Tensor -> list of shapes
            shape_list = [torch.Size(t.tolist()) for t in shape_list.unbind()]

        if len(shape_list) == 0:
            return [0]

        rank = len(shape_list[0])
        for s in shape_list:
            if len(s) != rank:
                raise RuntimeError(
                    "Cannot determine NT size for tensor components of inconsistent rank.")

        sizes = [len(shape_list)]
        for i in range(rank):
            dim_sizes = [shape[i] for shape in shape_list]
            if all([ds == dim_sizes[0] for ds in dim_sizes]):
                # Consistent value for this dim
                sizes.append(dim_sizes[0])
            else:
                sizes.append(dim_sizes)

        return sizes

    @classmethod
    def from_tensor_list(
            cls, tensor_list: List[Tensor], *, dtype=None, device=None, requires_grad=False):
        sizes = cls._sizes_from_shape_list([t.shape for t in tensor_list])
        buffer = (
            # TODO: Support pin_memory
            torch.cat([t.to(dtype=dtype, device=device).contiguous().view(-1)
                       for t in tensor_list]).requires_grad_(requires_grad)
            if len(tensor_list) > 0
            else torch.tensor([], dtype=dtype, device=device)
        )

        return cls(buffer, sizes).requires_grad_(requires_grad)

    def __repr__(self):
        prefix = "NestedTensor"
        if self.is_meta or isinstance(self.buffer, torch._subclasses.FakeTensor):
            return f"{prefix}(..., n_tensors={self.n_tensors}, dim={self.total_dim}, device=\"{self.device}\")"

        s = self.unbind()
        list_str = '\n'.join([str(t) for t in s])
        return f"{prefix}([\n{list_str}\n])"

    def dim(self):
        return self.total_dim

    # nt.shape calls here as well
    def size(self, dim=None):
        if dim is None:
            return self.sizes
        return self.sizes[dim]

    # Returns size for the ith component of the NT
    def _component_size(self, i):
        component_size = []
        for j in range(1, self.total_dim):
            if isinstance(self.sizes[j], list):
                component_size.append(self.sizes[j][i])
            else:
                component_size.append(self.sizes[j])
        return torch.Size(component_size)

    def unbind(self) -> List[torch.Tensor]:
        output = []
        offset = 0
        for i in range(self.n_tensors):
            tensor_size = self._component_size(i)
            numel = tensor_size.numel()
            output.append(self.buffer[offset:offset+numel].view(tensor_size))
            offset += numel
        return output

    def to_padded_tensor(self, padding, output_size=None) -> torch.Tensor:
        if self.n_tensors == 0:
            return self.buffer.clone()

        if output_size is None:
            output_size = [self.n_tensors]
            for i in range(self.total_dim-1):
                output_size.append(max([self._component_size(j)[i] for j in range(self.n_tensors)]))

        padded = self.buffer.new_empty(output_size).fill_(padding)
        for i, component in enumerate(self.unbind()):
            padded[i][tuple(slice(None, component.shape[j])
                       for j in range(self.total_dim-1))].copy_(component)

        return padded

    def __getattribute__(self, key):
        if key == "is_nested":
            return True
        if key == "shape":
            return self.size()

        # These are hacks to get around the fact that the NestedTensor itself is not a FakeTensor,
        # but simply contains a FakeTensor buffer.
        if key == "fake_mode":
            return self.buffer.fake_mode
        if key == "constant":
            return self.buffer.constant

        return super().__getattribute__(key)

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        print('DISPATCH:', func)

        # Call fallback. Currently assuming for simplicity that all arguments are NestedTensors
        def unwrap(t):
            return t.buffer if isinstance(t, NestedTensor) else t

        out = NestedTensor(
            func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs)),
            args[0].sizes)

        return out

    __torch_function__ = torch._C._disabled_torch_function_impl

class UnaryTest(TestCase):
    def test_multiplies(self):
        a = torch.rand(3, 4)
        b = torch.rand(8, 4)
        native_nt = torch.nested.as_nested_tensor([a, b])
        dispatch_nt = NestedTensor.from_tensor_list([a, b])
        result_native = native_nt * 2
        result_dispatch = dispatch_nt * 2
        for native, dispatch in zip(result_native.unbind(), result_dispatch.unbind()):
            self.assertEqual(native, dispatch)

class BinaryTest(TestCase):
    def test_addition(self):
        a = torch.rand(3, 4)
        b = torch.rand(8, 4)
        native_nt = torch.nested.as_nested_tensor([a, b])
        dispatch_nt = NestedTensor.from_tensor_list([a, b])
        result_native = native_nt + native_nt
        result_dispatch = dispatch_nt+ dispatch_nt
        for native, dispatch in zip(result_native.unbind(), result_dispatch.unbind()):
            self.assertEqual(native, dispatch)

if __name__ == "__main__":
    run_tests()
