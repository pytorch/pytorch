import torch

# A simple tensor subclass that holds two tensors internally, and runs every op on both tensors.
class DoubleTensor(torch.Tensor):

    @staticmethod
    def __new__(cls, a, b, *, requires_grad: bool):
        assert a.device == b.device and a.layout == b.layout and a.requires_grad == b.requires_grad and a.dtype == b.dtype
        # I guess it would be more accurate to represent the shape as torch.cat(a, b).shape
        shape = a.shape
        kwargs = {}
        kwargs["device"] = a.device
        kwargs["layout"] = a.layout
        kwargs["requires_grad"] = a.requires_grad
        kwargs["dtype"] = a.dtype
        return torch.Tensor._make_wrapper_subclass(
            cls, shape, **kwargs
        )

    def __init__(self, a, b, *, requires_grad: bool):
        self.a = a
        self.b = b
        self.requires_grad = requires_grad

    def __repr__(self):
        a_repr = repr(self.a)
        b_repr = repr(self.b)
        return f"DoubleTensor({a_repr}, {b_repr})"

    def __tensor_flatten__(self):
        return [self.a, self.b], self.requires_grad

    @staticmethod
    def __tensor_unflatten__(inner_tensors, meta):
        assert isinstance(meta, bool)
        a, b = inner_tensors
        return DoubleTensor(a, b, requires_grad=meta)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        if kwargs is None:
            kwargs = {}
        assert any(isinstance(x, DoubleTensor) for x in args)
        assert any(isinstance(x, DoubleTensor) for x in args)
        args_a = [x.a if isinstance(x, DoubleTensor) else x for x in args]
        args_b = [x.b if isinstance(x, DoubleTensor) else x for x in args]
        out_a = func(*args_a, **kwargs)
        out_b = func(*args_b, **kwargs)
        assert type(out_a) == type(out_b)
        # TODO: figure out the right way to propagate requires_grad-ness
        any_requires_grad = any(isinstance(x, torch.Tensor) and x.requires_grad for x in args)
        if isinstance(out_a, torch.Tensor):
            return DoubleTensor(out_a, out_b, requires_grad=any_requires_grad)
        # for aten ops that return non-tensors, just assume that
        # our two inner tensors return the same value
        assert out_a == out_b
        return out_a
