import torch
import torch.utils._pytree as pytree


class TensorSubclass(torch.Tensor):
    @staticmethod
    def __new__(cls, a):
        # I guess it would be more accurate to represent the shape as torch.cat(a, b).shape
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
        self.a = a

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        if kwargs is None:
            kwargs = {}

        print(f"func: {func}")

        args_a = pytree.tree_map_only(TensorSubclass, lambda x: x.a, args)

        kwargs_a = pytree.tree_map_only(TensorSubclass, lambda x: x.a, kwargs)

        out = TensorSubclass(func(*args_a, **kwargs_a))
        return out


class TensorSubclassMode(torch.utils._python_dispatch.TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **kwargs)
        if torch._subclasses.fake_tensor._is_tensor_constructor(func):
            out = TensorSubclass(out)
        return out


model = torch.nn.Linear(4, 4)
inputs = TensorSubclass(torch.randn(4, 4, requires_grad=True))

with TensorSubclassMode():
    out = model(inputs)

loss = out.sum()
print("fwd done")
loss.backward()
print("bwd done")
