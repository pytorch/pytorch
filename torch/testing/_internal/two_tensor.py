import torch
import torch.utils._pytree as pytree
from torch.utils._python_dispatch import return_and_correct_aliasing


# A simple tensor subclass that holds two tensors internally, and runs every op on both tensors.
class OneTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, a):
        # a = a[0]
        shape = a.shape
        kwargs = {}
        kwargs["strides"] = a.stride()
        kwargs["storage_offset"] = a.storage_offset()
        kwargs["device"] = a.device
        kwargs["layout"] = a.layout
        kwargs["requires_grad"] = a.requires_grad
        kwargs["dtype"] = a.dtype
        out = torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)
        return out

    def __init__(self, a):
        # a = a[0]
        self.a = a

    def __repr__(self):
        a_repr = repr(self.a)
        return f"OneTensor({a_repr})"

    def __tensor_flatten__(self):
        return ["a"], None

    @staticmethod
    def __tensor_unflatten__(inner_tensors, meta):
        assert meta is None
        a = inner_tensors["a"]
        return OneTensor(a)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        if kwargs is None:
            kwargs = {}
        args_a = pytree.tree_map_only(OneTensor, lambda x: x.a, args)

        kwargs_a = pytree.tree_map_only(OneTensor, lambda x: x.a, kwargs)

        print(f"here1")

        out_a = func(*args_a, **kwargs_a)
        print(f"out_a: {out_a}")
        print(f"here2")
        out_a_flat, spec = pytree.tree_flatten(out_a)
        print(f"here3")
        # for aten ops that return non-tensors, just assume that
        # our two inner tensors return the same value
        # breakpoint()
        out_flat = [
            OneTensor(o_a) if isinstance(o_a, torch.Tensor) else o_a
            for o_a in out_a_flat
        ]
        out = pytree.tree_unflatten(out_flat, spec)
        out_corrected = return_and_correct_aliasing(func, args, kwargs, out)

        # breakpoint()

        return out_corrected
