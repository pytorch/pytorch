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
    ragged_size: torch.SymInt
    is_fake: bool

    __torch_function__ = torch._C._disabled_torch_function_impl

    @staticmethod
    def __new__(
        cls,
        values,
        offsets,
        *,
        sym_size=None,
        raggedness_id=None,
        is_fake=False,
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
        # TODO: why is values requires grad?
        # if r.requires_grad:
        #     raise ValueError(
        #         "buffer should not require grad when constructing NestedTensor")
        r._values = values.detach() if values.requires_grad else values
        return r

    def __init__(self, values, offsets, *, sym_size=None, raggedness_id=None, **kwargs):
        super().__init__()
        # Only support jagged for now.
        assert offsets is not None
        assert offsets.ndim == 1
        assert not isinstance(values, NestedTensor)
        assert values.ndim == 2

        if sym_size is not None:
            # Passed during meta utils fakification
            self._size = sym_size
            self.raggedness_id = self._size[1]
        else:
            # We need to pass in raggedness id because in the situation where
            # raggedness id is symbolic, calling get_tensor_id with the same
            # offsets would not help us recover the symbolic int.
            if raggedness_id is not None:
                self.raggedness_id = raggedness_id
            else:
                self.raggedness_id = get_tensor_id(offsets)
            D = values.shape[1]
            B = offsets.shape[0] - 1
            self._size = (B, self.raggedness_id, D)
        self._offsets = offsets
        return

    def values(self):
        return self._values

    def offsets(self):
        return self._offsets

    def set_raggedness_id(self, id):
        self.raggedness_id = id
        self._size = (self._size[0], id, self._size[2])

    def __repr__(self):
        # We should implement this in torch/_tensor_str.py instead
        grad_fn_str = (
            f", requires_grad={self.requires_grad}" if self.requires_grad else ""
        )
        if self.grad_fn:
            grad_fn_str = f", grad_fn={self.grad_fn}"
        return f"NestedTensor(size={self._size}, offsets={self.offsets}{grad_fn_str})"

    def __tensor_flatten__(self):
        return ["_values", "_offsets"], (
            self.size(1),
            self.requires_grad,
        )

    @staticmethod
    def __tensor_unflatten__(inner_tensors: Dict, meta):
        assert len(inner_tensors) == 2
        values = inner_tensors["_values"]
        offsets = inner_tensors["_offsets"]
        (
            symint,
            requires_grad,
        ) = meta

        B = offsets.shape[0] - 1
        D = values.shape[1]
        sym_size = (B, symint, D)

        return NestedTensor(
            values, offsets=offsets, sym_size=sym_size, requires_grad=requires_grad
        )

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs

        # Lazy import to avoid circular dependency
        from .ops import lookup_jagged

        fn = lookup_jagged(func, *args, **kwargs)
        if fn is not None:
            return fn(*args, **kwargs)

        raise NotImplementedError


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
