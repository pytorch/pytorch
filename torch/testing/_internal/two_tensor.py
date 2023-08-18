import torch
from torch.utils._python_dispatch import return_and_correct_aliasing


# A simple tensor subclass that holds two tensors internally, and runs every op on both tensors.
class TwoTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, a, b):
        assert (
            a.device == b.device
            and a.layout == b.layout
            and a.requires_grad == b.requires_grad
            and a.dtype == b.dtype
        )
        # I guess it would be more accurate to represent the shape as torch.cat(a, b).shape
        shape = a.shape
        kwargs = {}
        kwargs["device"] = a.device
        kwargs["layout"] = a.layout
        kwargs["requires_grad"] = a.requires_grad
        kwargs["dtype"] = a.dtype
        out = torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)

        # If a and b are non-contiguous (or have weird storage offsets, etc)
        # Then we want to set the metadata on our wrapper properly too.
        assert a.shape == b.shape
        assert a.stride() == b.stride()
        assert a.storage_offset() == b.storage_offset()
        if (
            out.shape != a.shape
            or out.stride() != a.stride()
            or out.storage_offset() != a.storage_offset()
        ):
            with torch.utils._mode_utils.no_dispatch():
                out.as_strided_(a.shape, a.stride(), a.storage_offset())
        return out

    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __repr__(self):
        a_repr = repr(self.a)
        b_repr = repr(self.b)
        return f"TwoTensor({a_repr}, {b_repr})"

    def __tensor_flatten__(self):
        return ["a", "b"], None

    @staticmethod
    def __tensor_unflatten__(inner_tensors, meta):
        assert meta is None
        a, b = inner_tensors["a"], inner_tensors["b"]
        return TwoTensor(a, b)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        if kwargs is None:
            kwargs = {}
        assert any(isinstance(x, TwoTensor) for x in args)
        assert any(isinstance(x, TwoTensor) for x in args)
        args_a = [x.a if isinstance(x, TwoTensor) else x for x in args]
        args_b = [x.b if isinstance(x, TwoTensor) else x for x in args]
        out_a = func(*args_a, **kwargs)
        out_b = func(*args_b, **kwargs)
        assert type(out_a) == type(out_b)
        # TODO: figure out the right way to propagate requires_grad-ness
        any_requires_grad = any(
            isinstance(x, torch.Tensor) and x.requires_grad for x in args
        )
        if isinstance(out_a, torch.Tensor):
            out = TwoTensor(out_a, out_b)
            # Use this helper API to ensure that storage aliasing is correct. Needed for torch.compile correctness
            return return_and_correct_aliasing(func, args, kwargs, out)
        # for aten ops that return non-tensors, just assume that
        # our two inner tensors return the same value
        assert out_a == out_b
        return out_a
