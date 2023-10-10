from typing import Tuple

import torch
from torch._C import DispatchKey, DispatchKeySet
from torch.utils.weak import WeakTensorKeyDictionary
from typing import *  # noqa: F403

_tensor_id_counter = 0
_tensor_id_registry = WeakTensorKeyDictionary()


def get_tensor_id(tensor, *, coeff=1):
    global _tensor_id_counter
    if tensor not in _tensor_id_registry:
        _tensor_id_registry[tensor] = _tensor_id_counter
        _tensor_id_counter += 1
    return torch._C._get_singleton_int(_tensor_id_registry[tensor], coeff)


class NestedTensor(torch.Tensor):
    _values: torch.Tensor  # type: ignore[assignment]
    _offsets: torch.Tensor
    # NOTE [ Singleton ints for ragged sizes and strides ]
    #
    # Jagged layout tensors are tensors that represent a n-dim tensor with a
    # ragged dimension, but are backed by an (n-1)-dim tensor underneath, e.g.,
    # a jagged tensor with outer shape [B, x, D] is represented internally by a
    # tensor with shape [sum(x), D] where we introduce what we call a singleton
    # (or skolem) denoted as "x" here (but sometimes denoted with "*" to
    # represent the ragged dimension, and sum(x) represents the dim of the inner
    # tensor or equivalently the sum of all the sizes of the constituent
    # tensors' varying lengths.
    #
    # We also use singleton ints to represent the strides of this tensor.
    # For example, a jagged tensor with shape [B, x, D] can be strided in two
    # ways: [xD, D, 1] and [x, 1, sum(x)], where xD represents x multiplied by D
    #
    _size: Tuple[int, torch.SymInt, int]
    _stride: Tuple[torch.SymInt, int, int]
    # Indicates that the nth dimension is ragged
    _ragged_idx: int
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
            ragged_size = get_tensor_id(offsets, coeff=1)
        D = values.shape[1]
        B = offsets.shape[0] - 1
        # TODO: generalize for generalized raggedness
        self._size = (B, ragged_size, D)
        self._strides = (ragged_size * D, D, 1)
        self._ragged_idx = 1

        if values.requires_grad:
            raise ValueError(
                "NestedTensor values cannot require grad, please "
                "detach before passing to NestedTensor constructor"
            )
        self._values = values
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
        ctx = {
            "requires_grad": self.requires_grad,
            "ragged_size": self._size[self._ragged_idx],
        }
        return ["_values", "_offsets"], ctx

    @staticmethod
    def __tensor_unflatten__(inner_tensors: Dict, meta):
        assert len(inner_tensors) == 2
        values = inner_tensors["_values"]
        offsets = inner_tensors["_offsets"]

        return NestedTensor(
            values,
            offsets=offsets,
            ragged_size=meta["ragged_size"],
            requires_grad=meta["requires_grad"],
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
        return NestedTensor(values.detach(), offsets=offsets)

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
