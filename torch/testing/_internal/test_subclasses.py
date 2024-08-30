import torch
import torch.utils._pytree as pytree


class SubclassWithTensorFactory(torch.Tensor):
    @staticmethod
    def __new__(cls, src):
        shape = src.shape
        kwargs = {}
        kwargs["strides"] = src.stride()
        kwargs["storage_offset"] = src.storage_offset()
        kwargs["device"] = src.device
        kwargs["layout"] = src.layout
        kwargs["requires_grad"] = src.requires_grad
        kwargs["dtype"] = src.dtype
        out = torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)
        return out

    def __init__(self, src):
        self.src = src

    def __repr__(self):
        return f"{self.__class__.__name__}"

    def __tensor_flatten__(self):
        return ["src"], None

    @classmethod
    def __tensor_unflatten__(cls, inner_tensors, meta, outer_size, outer_stride):
        src = inner_tensors["src"]
        return cls(src)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        if kwargs is None:
            kwargs = {}

        _args = pytree.tree_map_only(
            cls, lambda x: x.src + torch.ones(x.src.shape), args
        )

        _kwargs = pytree.tree_map_only(
            cls, lambda x: x.src + torch.ones(x.src.shape), kwargs
        )

        _out = func(*_args, **_kwargs)

        _out_flat, _out_spec = pytree.tree_flatten(_out)

        out_flat = [cls(o) if isinstance(o, torch.Tensor) else o for o in _out_flat]
        return pytree.tree_unflatten(out_flat, _out_spec)
