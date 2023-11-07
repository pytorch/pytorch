from typing import Tuple

import torch
from torch._C import DispatchKey, DispatchKeySet
from torch._prims_common import is_expandable_to
from torch.fx.experimental.symbolic_shapes import free_symbols
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
    _lengths: Optional[torch.Tensor]
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
    _size: Tuple[int, ...]
    _stride: Tuple[int, ...]
    # Indicates that the nth dimension is ragged
    _ragged_idx: int

    @staticmethod
    def __new__(
        cls,
        values,
        offsets,
        *,
        lengths=None,
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
            torch.jagged,
            values.device,
            False,
            False,
            "sizes",
            False,
            True,  # dispatch_layout
            ks,
        )
        return r

    def __init__(self, values, offsets, *, lengths=None, ragged_size=None, **kwargs):
        super().__init__()
        # Only support jagged for now.
        assert offsets is not None
        assert offsets.ndim == 1
        assert not isinstance(values, NestedTensor)

        if ragged_size is None:
            # ragged_size needs to be explicitly passed during tracing (1) when
            # we initially fakify the nested tensor, and (2) when we rewrap as
            # we perform operations on fake nested tensors.
            # Calling get_tensor_id won't work in those cases because we want
            # the existing symbolic ragged_size to be propagated.
            if lengths is None:
                ragged_size = get_tensor_id(offsets, coeff=1)
            else:
                ragged_size = get_tensor_id(lengths, coeff=1)
        B = offsets.shape[0]
        Ds = values.shape[1:]
        self._size = (B, ragged_size, *Ds)
        stride = values.stride()
        self._strides = (ragged_size * stride[0], *stride)
        self._ragged_idx = 1

        if values.requires_grad:
            raise ValueError(
                "NestedTensor values cannot require grad, please "
                "detach before passing to NestedTensor constructor"
            )
        self._values = values
        self._offsets = offsets
        self._lengths = lengths

    def values(self):
        return self._values

    def offsets(self):
        return self._offsets

    def lengths(self):
        return self._lengths

    def __repr__(self):
        # We should implement this in torch/_tensor_str.py instead
        grad_fn_str = (
            f", requires_grad={self.requires_grad}" if self.requires_grad else ""
        )
        if self.grad_fn:
            grad_fn_str = f", grad_fn={self.grad_fn}"
        return f"NestedTensor(size={self._size}, offsets={self._offsets}{grad_fn_str}, contiguous={self._lengths is None})"

    def __reduce_ex__(self, proto):
        state = torch._utils._get_obj_state(self)

        # SymNodes are not serializable
        assert "_size" in state and "_strides" in state
        state = dict(state)
        del state["_size"]
        del state["_strides"]

        func = NestedTensor
        args = (self._values, self._offsets)
        return (torch._tensor._rebuild_from_type_v2, (func, type(self), args, state))

    def __tensor_flatten__(self):
        ctx = {
            "requires_grad": self.requires_grad,
            "ragged_size": self._size[self._ragged_idx],
        }
        inner_tensors = ["_values", "_offsets"]
        if self._lengths is not None:
            inner_tensors.append("_lengths")
        return inner_tensors, ctx

    @staticmethod
    def __tensor_unflatten__(inner_tensors: Dict, meta):
        assert len(inner_tensors) >= 2 and len(inner_tensors) <= 3
        values = inner_tensors["_values"]
        offsets = inner_tensors["_offsets"]
        if "_lengths" in inner_tensors and inner_tensors["_lengths"] is not None:
            lengths = inner_tensors["_lengths"]
        else:
            lengths = None

        # NOTE [ Storing symbolic values as plain attributes on subclasses ]
        #
        # When a subclass like NestedTensor stores a "size-like" value (which
        # can either be Symintified or not) into meta, it's responsible for:
        #
        #   (1) Propagating that symint during torch dispatch when performing
        #       operations, i.e. torch dispatch plays the role of a meta kernel.
        #
        #   (2) Facilitating the behavior around symbolic -> non-symbolic
        #       conversions and vice versa, see below.
        #
        # [ non-symbolic -> symbolic (fakification in meta_utils) ]
        #
        # __tensor_unflatten__ is passed symbolic dense tensors and meta from
        # non-symbolic subclasses. In this case, the subclass is responsible for
        # intercepting meta["ragged_size"] for example and replacing it with the
        # symintified version.
        #
        # [ symbolic -> non-symbolic ]
        #
        # __tensor_unflatten__ is passed non-symbolic dense tensors and with
        # meta extracted from fake subclasses. In this case the subclass gets
        # propagated the meta["ragged_size"] which is still a symint and the
        # subclass is responsible for making sure that the symint doesn't leak.
        #
        if len(free_symbols(values)) == 0 and len(free_symbols(offsets)) == 0:
            # Note that we cannot simply check if is_fake(values) because
            # during aot autograd, FunctionalTensors are not fake but hold
            # symbolic sizes.
            meta["ragged_size"] = None

        return NestedTensor(
            values,
            offsets=offsets,
            lengths=lengths,
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

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        from .ops import jagged_torch_function

        try:
            return jagged_torch_function(func, *args, **kwargs)
        except NotImplementedError:
            pass
        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)


# Not actually a view!
class ViewBufferFromNested(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: NestedTensor):  # type: ignore[override]
        ctx.save_for_backward(x.offsets())
        ctx.kwargs = {
            "ragged_size": x._size[x._ragged_idx],
        }
        return x.values()

    @staticmethod
    def backward(ctx, gO: torch.Tensor):  # type: ignore[override]
        (offsets,) = ctx.saved_tensors
        return NestedTensor(gO, offsets=offsets, **ctx.kwargs)


# Not actually a view!
class ViewNestedFromBuffer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values: torch.Tensor, offsets: torch.Tensor):  # type: ignore[override]
        return NestedTensor(values.detach(), offsets=offsets)

    @staticmethod
    def backward(ctx, gO: NestedTensor):  # type: ignore[override]
        return gO.values(), None, None


# Not actually a view!
class ViewNonContiguousNestedFromBuffer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values: torch.Tensor, offsets: torch.Tensor, lengths: torch.Tensor):  # type: ignore[override]
        return NestedTensor(values.detach(), offsets=offsets, lengths=lengths)

    @staticmethod
    def backward(ctx, gO: NestedTensor):  # type: ignore[override]
        return gO.values(), None, None


# Need to make it obvious that users should be passing in offsets
def jagged_from_list(
    tensors: List[torch.Tensor],
    offsets: Optional[torch.Tensor],
    dtype=None,
    device=None,
) -> Tuple[NestedTensor, torch.Tensor]:
    """Constructs a NestedTensor backed by jagged layout from a list of tensors"""

    if not len(set(t.dtype for t in tensors)) == 1:  # noqa: C401
        raise RuntimeError(
            "When constructing a nested tensor, all tensors in list must have the same dtype"
        )
    if not len(set(t.device for t in tensors)) == 1:  # noqa: C401
        raise RuntimeError(
            "When constructing a nested tensor, all tensors in list must be on the same device"
        )

    # Check that the NT is representable by the jagged layout.
    # Jagged layout represents (B, *, D_0, D_1, ..., D_N), where the only
    # raggedness allowed is for the single dim immediately adjacent to the batch dim.
    sizes = [t.shape for t in tensors]
    non_first_sizes = [s[1:] for s in sizes]
    at_most_first_ragged = all(s == non_first_sizes[0] for s in non_first_sizes)
    if not at_most_first_ragged:
        raise RuntimeError(
            "Cannot represent given tensor list as a nested tensor with the jagged layout. "
            "Note that the jagged layout only represents shapes of the form "
            "(B, *, D_0, D_1, ..., D_N), with only * allowed to be ragged."
        )

    # Set properties appropriately.
    values = torch.cat(tensors, dim=0)
    to_kwargs = {}
    if device is not None:
        to_kwargs["device"] = device
    if dtype is not None:
        to_kwargs["dtype"] = dtype
    values = values.to(**to_kwargs)

    # Calculate jagged offsets if not provided.
    if offsets is None:
        # Jagged layout specifies that offsets are stored as int64 on the same device as values.
        offsets = torch.cat(
            [
                torch.zeros(1, dtype=torch.int64, device=values.device),
                torch.tensor([s[0] for s in sizes[:-1]], device=values.device).cumsum(
                    dim=0
                ),
            ]
        )

    return ViewNestedFromBuffer.apply(values, offsets), offsets  # type: ignore[call-overload]


def jagged_from_tensor_and_lengths(
    tensor: torch.Tensor, starts: torch.Tensor, lengths: torch.Tensor
) -> Tuple[NestedTensor, torch.Tensor, torch.Tensor]:
    """Constructs a NestedTensor backed by jagged layout from a tensor, starts of sequences, and sequence lengths"""
    batch_size = tensor.shape[0]
    if is_expandable_to(starts.shape, (batch_size,)) and is_expandable_to(
        lengths.shape, (batch_size,)
    ):
        start_list = starts.expand(batch_size)
        length_list = lengths.expand(batch_size)
    else:
        raise RuntimeError(
            "When constructing a jagged nested tensor using narrow(), "
            "your start and length must be a Tensor that broadcasts to input.shape[0] x 1"
        )

    # Calculate jagged offsets
    assert (
        len(tensor.shape) >= 2
    ), "tensor must at least be 2D for the nested narrow op to work"
    max_seq_len = tensor.shape[1]
    offset_lengths = max_seq_len * torch.arange(
        0, batch_size, dtype=torch.int64, device=tensor.device
    )
    # Jagged layout specifies that offsets are stored as int64 on the same device as values.
    offsets = start_list + offset_lengths

    # Reshape buffer to flatten the 1st and 2nd dimension (view used to enforce non-copy)
    if len(tensor.shape) > 2:
        values = tensor.view(-1, *tensor.shape[2:])
    else:
        values = tensor.view(-1)

    return ViewNonContiguousNestedFromBuffer.apply(values, offsets, length_list), offsets, length_list  # type: ignore[call-overload]


def buffer_from_jagged(jagged):
    return ViewBufferFromNested.apply(jagged)
