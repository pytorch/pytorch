from typing import Tuple

import torch
from torch._C import DispatchKey, DispatchKeySet
from torch.utils.weak import WeakTensorKeyDictionary
from typing import *  # noqa: F403

_tensor_id_counter = 0
_tensor_id_registry = WeakTensorKeyDictionary()


def get_tensor_id(tensor, coeff):
    global _tensor_id_counter
    if tensor not in _tensor_id_registry:
        _tensor_id_registry[tensor] = _tensor_id_counter
        _tensor_id_counter += 1
    return torch._C._get_singleton_int(_tensor_id_registry[tensor], coeff)


class NestedTensor(torch.Tensor):
    _values: torch.Tensor  # type: ignore[assignment]
    _offsets: torch.Tensor
    _size: Tuple[int, torch.SymInt, int]
    # TODO: Write a note here about what how we are using Singleton ints for
    # sizes and strides.
    _ragged_size: torch.SymInt

    __torch_function__ = torch._C._disabled_torch_function_impl

    @staticmethod
    def __new__(
        cls,
        values,
        offsets,
        *,
        ragged_size=None,
        **kwargs,
    ):
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
        return r

    def __init__(self, values, offsets, *, ragged_size=None, **kwargs):
        super().__init__()
        # Only support jagged for now.
        assert offsets is not None
        assert offsets.ndim == 1
        assert not isinstance(values, NestedTensor)
        assert values.ndim == 2

        if ragged_size is None:
            # ragged_size needs to be explicitly passed during tracing (1) when
            # we initially fakify the nested tensor, and (2) when we rewrap as
            # we perform operations on fake nested tensors.
            # Calling get_tensor_id won't work in those cases because we want
            # the existing symbolic ragged_size to be propagated.
            ragged_size = get_tensor_id(offsets, 1)
        D = values.shape[1]
        B = offsets.shape[0] - 1
        # TODO: factor out and generalize the stride computing logic
        self._size = (B, ragged_size, D)
        self._strides = (ragged_size * D, D, 1)
        self._ragged_size = ragged_size

        # TODO: error if values requires grad
        self._values = values.detach() if values.requires_grad else values
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
        return f"NestedTensor(size={self._size}, offsets={self.offsets}{grad_fn_str})"

    def __tensor_flatten__(self):
        return ["_values", "_offsets"], (self.requires_grad, self._ragged_size)

    @staticmethod
    def __tensor_unflatten__(inner_tensors: Dict, meta):
        assert len(inner_tensors) == 2
        values = inner_tensors["_values"]
        offsets = inner_tensors["_offsets"]
        (requires_grad, ragged_size, *extra_meta) = meta

        if len(extra_meta) > 0:
            # During fakification, the ragged_size is passed in as extra context
            # because we need its symint version.
            (ragged_size,) = extra_meta

        return NestedTensor(
            values,
            offsets=offsets,
            ragged_size=ragged_size,
            requires_grad=requires_grad,
        )

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs

        # Lazy import to avoid circular dependency
        from .ops import lookup_jagged

        fn = lookup_jagged(func, *args, **kwargs)
        if fn is not None:
            return fn(*args, **kwargs)

        raise NotImplementedError(func)


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
    assert all(t.ndim == 2 for t in tensors)
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
