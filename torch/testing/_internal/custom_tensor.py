# mypy: ignore-errors


from collections import namedtuple

import torch
import torch.utils._pytree as pytree
from torch.utils._python_dispatch import return_and_correct_aliasing


FancyNamedTuple = namedtuple("FancyNamedTuple", ["foo", "bar"])


# A simple tensor subclass that holds a tensor with custom metadata and custom method
class ConstantExtraMetadataTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, elem):
        shape = elem.shape
        kwargs = {}
        kwargs["strides"] = elem.stride()
        kwargs["storage_offset"] = elem.storage_offset()
        kwargs["device"] = elem.device
        kwargs["layout"] = elem.layout
        kwargs["requires_grad"] = elem.requires_grad
        kwargs["dtype"] = elem.dtype
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)

    def __init__(self, elem):
        self.elem = elem
        self.constant_attribute = 4

    def __repr__(self):
        inner_repr = repr(self.elem)
        return f"CustomTensor({inner_repr})"

    def get_complicated_metadata(self):
        return FancyNamedTuple(self.constant_attribute, self.constant_attribute)

    def __tensor_flatten__(self):
        return ["elem"], self.constant_attribute

    def add_constant(self, a):
        self.constant_attribute += a

    @staticmethod
    def __tensor_unflatten__(inner_tensors, meta, outer_size, outer_stride):
        assert meta is not None
        elem = inner_tensors["elem"]
        out = ConstantExtraMetadataTensor(elem)
        out.constant_attribute = meta
        return out

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        if kwargs is None:
            kwargs = {}
        args_inner = pytree.tree_map_only(
            ConstantExtraMetadataTensor, lambda x: x.elem, args
        )

        kwargs_inner = pytree.tree_map_only(
            ConstantExtraMetadataTensor, lambda x: x.elem, kwargs
        )

        out_inner = func(*args_inner, **kwargs_inner)
        out_inner_flat, spec = pytree.tree_flatten(out_inner)
        # for aten ops that return non-tensors, just assume that
        # our cust inner tensors return the same value
        out_flat = [
            ConstantExtraMetadataTensor(o_inner)
            if isinstance(o_inner, torch.Tensor)
            else o_inner
            for o_inner in out_inner_flat
        ]
        out = pytree.tree_unflatten(out_flat, spec)
        return return_and_correct_aliasing(func, args, kwargs, out)


# A simple tensor subclass that always returns plain tensor during __torch_dispatch__
# It is similar to TwoTensor and is used to simulate torchao quantized tensors
class CustomTensorPlainOut(torch.Tensor):
    @staticmethod
    def __new__(cls, elem1, elem2):
        shape = elem1.shape
        kwargs = {}
        kwargs["strides"] = elem1.stride()
        kwargs["storage_offset"] = elem1.storage_offset()
        kwargs["device"] = elem1.device
        kwargs["layout"] = elem1.layout
        kwargs["requires_grad"] = elem1.requires_grad
        kwargs["dtype"] = elem1.dtype
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)

    def __init__(self, elem1, elem2):
        self.elem1 = elem1
        self.elem2 = elem2

    def get_elem(self):
        return self.elem1

    def __repr__(self):
        inner_repr_1 = repr(self.elem1)
        inner_repr_2 = repr(self.elem2)
        return f"CustomTensorPlainOut({inner_repr_1}, {inner_repr_2})"

    def __tensor_flatten__(self):
        return ["elem1", "elem2"], None

    @staticmethod
    def __tensor_unflatten__(inner_tensors, meta, outer_size, outer_stride):
        elem1 = inner_tensors["elem1"]
        elem2 = inner_tensors["elem2"]
        out = CustomTensorPlainOut(elem1, elem2)
        return out

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        # Don't use this tensor with view ops
        if kwargs is None:
            kwargs = {}
        args_inner_1 = pytree.tree_map_only(
            CustomTensorPlainOut, lambda x: x.elem1, args
        )

        kwargs_inner_1 = pytree.tree_map_only(
            CustomTensorPlainOut, lambda x: x.elem1, kwargs
        )

        args_inner_2 = pytree.tree_map_only(
            CustomTensorPlainOut, lambda x: x.elem2, args
        )

        kwargs_inner_2 = pytree.tree_map_only(
            CustomTensorPlainOut, lambda x: x.elem2, kwargs
        )

        out_inner_1 = func(*args_inner_1, **kwargs_inner_1)
        out_inner_2 = func(*args_inner_2, **kwargs_inner_2)

        out_inner_flat_1, spec = pytree.tree_flatten(out_inner_1)
        out_inner_flat_2, spec = pytree.tree_flatten(out_inner_2)

        if func.is_view:
            new_out = pytree.tree_unflatten(
                (
                    CustomTensorPlainOut(tensor1, tensor2)
                    for tensor1, tensor2 in zip(out_inner_flat_1, out_inner_flat_2)
                ),
                spec,
            )
            return return_and_correct_aliasing(func, args, kwargs, new_out)

        out_new = (
            out_inner_flat_1[ix] + out_inner_flat_2[ix]
            for ix in range(len(out_inner_flat_1))
        )

        return pytree.tree_unflatten(out_new, spec)
