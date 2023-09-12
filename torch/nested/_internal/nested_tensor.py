from typing import Tuple

import torch
from torch._C import DispatchKey, DispatchKeySet
from torch.utils.weak import WeakTensorKeyDictionary
from typing import *  # noqa: F403

_tensor_id_counter = 0
_tensor_id_registry = WeakTensorKeyDictionary()


def get_tensor_id(tensor):
    global _tensor_id_counter
    if tensor not in _tensor_id_registry:
        _tensor_id_registry[tensor] = _tensor_id_counter
        _tensor_id_counter += 1
    return torch._C._get_singleton_int(_tensor_id_registry[tensor])


class NestedTensor(torch.Tensor):
    _values: torch.Tensor  # type: ignore[assignment]
    _offsets: torch.Tensor
    _size: Tuple[int, int, int]

    __torch_function__ = torch._C._disabled_torch_function_impl

    @staticmethod
    def __new__(cls, values, offsets, **kwargs):
        ks = DispatchKeySet(DispatchKey.NestedTensor)
        ks = ks.add(DispatchKey.AutogradNestedTensor)
        r = torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
            cls,
            (0,),
            (0,),
            0,
            torch.contiguous_format,
            values.dtype,
            values.layout,
            values.device,
            False,
            False,
            "sizes",
            False,
            False,
            ks,
        )
        # TODO: why is values requires grad?
        # if r.requires_grad:
        #     raise ValueError(
        #         "buffer should not require grad when constructing NestedTensor")
        r._values = values.detach() if values.requires_grad else values
        return r

    def __init__(self, values, offsets, **kwargs):
        super().__init__()
        # Only support jagged for now.
        assert offsets is not None
        assert offsets.ndim == 1
        assert not isinstance(values, NestedTensor)

        # In a later PR, we'll need to accept an additional size argument
        # to handle dynamic shapes.
        ragged_dim = get_tensor_id(offsets)
        Ds = values.shape[1:]
        B = offsets.shape[0] - 1
        self._size = (B, ragged_dim, *Ds)
        self._offsets = offsets
        return

    def values(self):
        return self._values

    def offsets(self):
        return self._offsets

    def __repr__(self):
        # We should implement this in torch/_tensor_str.py instead
        grad_fn_str = (
            f", requires_grad={self.requires_grad}" if self.requires_grad else ""
        )
        if self.grad_fn:
            grad_fn_str = f", grad_fn={self.grad_fn}"
        return f"NestedTensor(size={self._size}, offsets={self._offsets}{grad_fn_str})"

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs

        # Lazy import to avoid circular dependency
        from .ops import lookup_jagged

        fn = lookup_jagged(func, *args, **kwargs)
        if fn is not None:
            return fn(*args, **kwargs)

        raise NotImplementedError(f"{str(func)}")

# Not actually a view!
class ViewBufferFromNested(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: NestedTensor):  # type: ignore[override]
        ctx.save_for_backward(x.offsets())
        return x.values()

    @staticmethod
    def backward(ctx, gO: torch.Tensor):  # type: ignore[override]
        (offsets,) = ctx.saved_tensors
        return NestedTensor(gO, offsets=offsets)


# Not actually a view!
class ViewNestedFromBuffer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values: torch.Tensor, offsets: torch.Tensor):  # type: ignore[override]
        return NestedTensor(values, offsets=offsets)

    @staticmethod
    def backward(ctx, gO: NestedTensor):  # type: ignore[override]
        return gO.values(), None, None


# Need to make it obvious that users should be passing in offsets
def jagged_from_list(
    tensors: Sequence[torch.Tensor], offsets: Optional[torch.Tensor]
) -> Tuple[NestedTensor, torch.Tensor]:
    """Constructs a NestedTensor backed by jagged layout from a list of tensors"""
    assert len(set(t.dtype for t in tensors)) == 1  # noqa: C401
    assert len(set(t.device for t in tensors)) == 1  # noqa: C401
    # TODO: Expand this check across all dims but the first
    assert len(set(t.shape[1] for t in tensors)) == 1  # noqa: C401

    lengths = torch.tensor([t.shape[0] for t in tensors])
    _offsets = torch.cat([torch.tensor([0]), lengths.cumsum(0)])
    if offsets is not None:
        assert torch.all(offsets == _offsets).item()
    else:
        offsets = _offsets

    return ViewNestedFromBuffer.apply(torch.cat(tensors, dim=0), offsets), offsets  # type: ignore[call-overload]


def buffer_from_jagged(jagged):
    return ViewBufferFromNested.apply(jagged)
