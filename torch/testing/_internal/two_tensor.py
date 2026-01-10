# mypy: ignore-errors

import torch
import torch.utils._pytree as pytree
from torch._export.wrappers import mark_subclass_constructor_exportable_experimental
from torch.utils._python_dispatch import return_and_correct_aliasing


# A simple tensor subclass that holds two tensors internally, and runs every op on both tensors.
class TwoTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, a, b, outer_size=None, outer_stride=None, *, requires_grad=None):
        if outer_size is None:
            outer_size = a.size()
        if outer_stride is None:
            outer_stride = a.stride()

        if (
            a.device != b.device
            or a.layout != b.layout
            or a.requires_grad != b.requires_grad
            or a.dtype != b.dtype
        ):
            raise AssertionError(
                "Inner tensors a and b must have matching device, layout, requires_grad, and dtype"
            )
        # I guess it would be more accurate to represent the shape as torch.cat(a, b).shape
        shape = outer_size
        kwargs = {}
        kwargs["strides"] = outer_stride
        kwargs["storage_offset"] = a.storage_offset()
        kwargs["device"] = a.device
        kwargs["layout"] = a.layout
        kwargs["requires_grad"] = requires_grad or a.requires_grad
        kwargs["dtype"] = a.dtype
        out = torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)

        if a.shape != b.shape:
            raise AssertionError(
                f"Tensors must have same shape: a.shape={a.shape}, b.shape={b.shape}"
            )
        if a.stride() != b.stride():
            raise AssertionError(
                f"Tensors must have same stride: a.stride={a.stride()}, b.stride={b.stride()}"
            )
        if a.storage_offset() != b.storage_offset():
            raise AssertionError(
                f"Tensors must have same storage_offset: a={a.storage_offset()}, b={b.storage_offset()}"
            )
        return out

    @torch._disable_dynamo
    @mark_subclass_constructor_exportable_experimental
    def __init__(self, a, b, outer_size=None, outer_stride=None, *, requires_grad=None):
        self.a = a
        self.b = b

    def __repr__(self):
        a_repr = repr(self.a)
        b_repr = repr(self.b)
        return f"TwoTensor({a_repr}, {b_repr})"

    def __tensor_flatten__(self):
        return ["a", "b"], None

    @staticmethod
    def __tensor_unflatten__(inner_tensors, meta, outer_size, outer_stride):
        if meta is not None:
            raise AssertionError("meta must be None for TwoTensor")
        a, b = inner_tensors["a"], inner_tensors["b"]
        if type(a) is torch.Tensor:
            if outer_size is None:
                raise AssertionError("outer_size must not be None when a is a Tensor")
            if outer_stride is None:
                raise AssertionError("outer_stride must not be None when a is a Tensor")
        return TwoTensor(a, b, outer_size, outer_stride)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        if kwargs is None:
            kwargs = {}
        args_a = pytree.tree_map_only(TwoTensor, lambda x: x.a, args)
        args_b = pytree.tree_map_only(TwoTensor, lambda x: x.b, args)

        kwargs_a = pytree.tree_map_only(TwoTensor, lambda x: x.a, kwargs)
        kwargs_b = pytree.tree_map_only(TwoTensor, lambda x: x.b, kwargs)

        out_a = func(*args_a, **kwargs_a)
        out_b = func(*args_b, **kwargs_b)
        out_a_flat, spec = pytree.tree_flatten(out_a)
        out_b_flat = pytree.tree_leaves(out_b)
        # for aten ops that return non-tensors, just assume that
        # our two inner tensors return the same value
        out_flat = [
            cls(o_a, o_b) if isinstance(o_a, torch.Tensor) else o_a
            for o_a, o_b in zip(out_a_flat, out_b_flat, strict=True)
        ]
        out = pytree.tree_unflatten(out_flat, spec)
        from torch._higher_order_ops.cond import cond_op

        if func is cond_op:
            return out
        else:
            return return_and_correct_aliasing(func, args, kwargs, out)

    def get_elem_a(self):
        return self.a


class TwoTensorMode(torch.utils._python_dispatch.TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **kwargs)
        if torch._subclasses.fake_tensor._is_tensor_constructor(func):
            out = TwoTensor(out, out.clone())
        return out
