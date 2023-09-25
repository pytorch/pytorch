from typing import Tuple

import torch
from torch._C import DispatchKey, DispatchKeySet
from torch.utils.weak import WeakTensorKeyDictionary
from typing import *  # noqa: F403

_tensor_id_counter = 0
_tensor_id_registry = WeakTensorKeyDictionary()


def get_tensor_id(tensor, factor):
    global _tensor_id_counter
    if tensor not in _tensor_id_registry:
        _tensor_id_registry[tensor] = _tensor_id_counter
        _tensor_id_counter += 1
    return torch._C._get_singleton_int(_tensor_id_registry[tensor], factor)


class NestedTensor(torch.Tensor):
    _values: torch.Tensor  # type: ignore[assignment]
    _offsets: torch.Tensor
    _size: Tuple[int, torch.SymInt, int]
    ragged_size: torch.SymInt

    __torch_function__ = torch._C._disabled_torch_function_impl

    @staticmethod
    def __new__(
        cls,
        values,
        offsets,
        *,
        sym_size=None,
        sym_stride=None,
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
        # TODO: why is values requires grad?
        # if r.requires_grad:
        #     raise ValueError(
        #         "buffer should not require grad when constructing NestedTensor")
        r._values = values.detach() if values.requires_grad else values
        return r

    def __init__(self, values, offsets, *, sym_size=None, sym_stride=None, **kwargs):
        super().__init__()
        # Only support jagged for now.
        assert offsets is not None
        assert offsets.ndim == 1
        assert not isinstance(values, NestedTensor)
        assert values.ndim == 2

        if sym_size is not None:
            assert sym_stride is not None
            # sym_size and sym_stride are passed during tracing (1) when we
            # initially fakify the nested tensor, and (2) when we rewrap as we
            # perform operations on fake nested tensors.
            self._size = sym_size
            self._strides = sym_stride
            self.ragged_size = self._size[1]  # type: ignore[assignment]
        else:
            self.ragged_size = get_tensor_id(offsets, 1)
            D = values.shape[1]
            B = offsets.shape[0] - 1
            # TODO: factor out and generalize the stride computing logic
            self._size = (B, self.ragged_size, D)
            self._strides = (self.ragged_size * D, D, 1)
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
        return ["_values", "_offsets"], (self.requires_grad, self.ragged_size)

    @staticmethod
    def __tensor_unflatten__(inner_tensors: Dict, meta):
        assert len(inner_tensors) == 2
        values = inner_tensors["_values"]
        offsets = inner_tensors["_offsets"]
        (requires_grad, ragged_size, *extra_meta) = meta

        if len(extra_meta) > 0:
            (source,) = extra_meta
            # Avoid circular import
            from torch._dynamo.source import TensorProperty, TensorPropertySource

            shape_env = offsets.shape[0].node.shape_env
            sym_ragged_size = shape_env.create_symintnode(
                shape_env.create_symbol(
                    ragged_size,
                    TensorPropertySource(source, TensorProperty.SIZE, 1),
                ),
                hint=ragged_size,
            )
            ragged_size = sym_ragged_size

        B = offsets.shape[0] - 1
        D = values.shape[1]
        sym_size = (B, ragged_size, D)
        # Assume contiguous
        sym_stride = (ragged_size * D, D, 1)

        return NestedTensor(
            values,
            offsets=offsets,
            sym_size=sym_size,
            sym_stride=sym_stride,
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
