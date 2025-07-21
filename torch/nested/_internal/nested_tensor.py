# mypy: allow-untyped-defs
from typing import *  # noqa: F403

import torch
from torch._C import DispatchKey, DispatchKeySet
from torch._prims_common import is_expandable_to
from torch.nested._internal.nested_int import NestedIntNode
from torch.utils.weak import WeakTensorKeyDictionary


_tensor_id_counter = 0
_tensor_symint_registry = WeakTensorKeyDictionary()


def get_tensor_symint(tensor, *, coeff=1):
    from torch._subclasses.fake_tensor import FakeTensor
    from torch._subclasses.functional_tensor import mb_unwrap_functional_tensor

    # NB: Only FakeTensor is associated with a memo
    tensor = mb_unwrap_functional_tensor(tensor)
    if isinstance(tensor, FakeTensor):
        return tensor.get_nested_int(coeff=coeff)

    global _tensor_id_counter

    tensor_symint = _tensor_symint_registry.get(tensor)
    if tensor_symint is None:
        tensor_symint = torch.SymInt(NestedIntNode(_tensor_id_counter, coeff))
        _tensor_id_counter += 1
        _tensor_symint_registry[tensor] = tensor_symint
    return tensor_symint


# SDPA metadata; max / min seqlens are needed for e.g. flash
def _get_sdpa_extreme_seqlen(func, tensor):
    return int(func(tensor).item())


def _store_val_in_tensor(val) -> torch.Tensor:
    # hack to get dynamic shapes support: store in a (val, 0) shaped tensor
    return torch.zeros(val, 0)


def _load_val_from_tensor(t: torch.Tensor):
    return t.shape[0]


# serialization function must be defined at top level
def _rebuild_njt(constructor_kwargs):
    return NestedTensor(**constructor_kwargs)


class NestedTensor(torch.Tensor):
    _values: torch.Tensor  # type: ignore[assignment]
    _offsets: torch.Tensor
    _lengths: Optional[torch.Tensor]
    # NOTE [ Nested ints for ragged sizes and strides ]
    #
    # Jagged layout tensors are tensors that represent a n-dim tensor with a
    # ragged dimension, but are backed by an (n-1)-dim tensor underneath, e.g.,
    # a jagged tensor with outer shape [B, x, D] is represented internally by a
    # tensor with shape [sum(x), D] where we introduce what we call a nested int
    # denoted as "x" here (but sometimes denoted with "*" to
    # represent the ragged dimension, and sum(x) represents the dim of the inner
    # tensor or equivalently the sum of all the sizes of the constituent
    # tensors' varying lengths.
    #
    # We also use nested ints to represent the strides of this tensor.
    # For example, a jagged tensor with shape [B, x, D] can be strided in two
    # ways: [xD, D, 1] and [x, 1, sum(x)], where xD represents x multiplied by D
    _size: tuple[int, ...]
    _strides: tuple[int, ...]
    # Indicates that the nth dimension is ragged
    _ragged_idx: int
    _metadata_cache: Dict[str, Any]

    @staticmethod
    def __new__(
        cls,
        values,
        offsets,
        *,
        lengths=None,
        **kwargs,
    ):
        ks = DispatchKeySet(DispatchKey.NestedTensor)
        ks = ks.add(DispatchKey.AutogradNestedTensor)

        # Only support jagged for now.
        assert offsets is not None
        assert offsets.ndim == 1
        assert not isinstance(values, NestedTensor)
        assert values.device == offsets.device

        # Query cache for the symint associated with offsets or lengths
        # (create a new one if needed).
        ragged_source = offsets if lengths is None else lengths
        ragged_size = get_tensor_symint(ragged_source, coeff=1)
        _ragged_idx = kwargs.get("_ragged_idx", 1)
        B = offsets.shape[0] - 1
        if lengths is not None:
            assert B == lengths.shape[0]

        # subtract 1 to convert to values dim space
        r = _ragged_idx - 1
        _size = (B, *values.shape[:r], ragged_size, *values.shape[r + 1 :])
        stride = values.stride()
        _strides = (ragged_size * stride[r], *stride)

        r = torch.Tensor._make_wrapper_subclass(
            cls,
            _size,
            _strides,
            0,
            torch.contiguous_format,
            values.dtype,
            torch.jagged,
            values.device,
            False,
            kwargs.get("requires_grad", False),
            "sizes",
            False,
            True,  # dispatch_layout
            ks,
            # don't try to calculate storage based on non-zero size
            storage_size=values.untyped_storage().size(),
        )
        r._ragged_idx = _ragged_idx
        r._size = _size
        r._strides = _strides

        return r

    def __init__(self, values, offsets, *, lengths=None, **kwargs):
        super().__init__()

        self._values = values
        self._offsets = offsets
        self._lengths = lengths

        # holds properties that are computed lazily
        self._metadata_cache = kwargs.get("_metadata_cache") or {}

        # collapsed ragged dim must always be dynamic
        torch._dynamo.maybe_mark_dynamic(self, self._ragged_idx)
        torch._dynamo.maybe_mark_dynamic(self._values, self._ragged_idx - 1)

        # min / max sequence length should be dynamic if present
        max_seqlen_tensor = self._metadata_cache.get("max_seqlen", None)
        if max_seqlen_tensor is not None:
            torch._dynamo.mark_dynamic(max_seqlen_tensor, 0)
        min_seqlen_tensor = self._metadata_cache.get("min_seqlen", None)
        if min_seqlen_tensor is not None:
            torch._dynamo.mark_dynamic(min_seqlen_tensor, 0)

    def values(self):
        # dispatch to get proper view relationship
        return torch._nested_get_values(self)  # type: ignore[attr-defined]

    def offsets(self):
        return self._offsets

    def lengths(self):
        return self._lengths

    # Private accessor functions for min / max sequence length. They're
    # purposefully not @properties because those don't work with PT2 (yet).
    # These compute / cache if not present.
    # TODO: Revisit this when @properties are better supported by PT2. I think the ideal
    # state would be to have public @properties for min / max sequence length that compile
    # (including setters).
    def _get_max_seqlen(self):
        max_seqlen_tensor = self._max_seqlen_tensor
        if max_seqlen_tensor is None:
            # compute & cache
            max_val = _get_sdpa_extreme_seqlen(
                torch.max,
                self._offsets.diff() if self._lengths is None else self._lengths,
            )
            max_seqlen_tensor = _store_val_in_tensor(max_val)
            self._metadata_cache["max_seqlen"] = max_seqlen_tensor
        return _load_val_from_tensor(max_seqlen_tensor)

    def _get_min_seqlen(self):
        min_seqlen_tensor = self._min_seqlen_tensor
        if min_seqlen_tensor is None:
            # compute & cache
            min_val = _get_sdpa_extreme_seqlen(
                torch.min,
                self._offsets.diff() if self._lengths is None else self._lengths,
            )
            min_seqlen_tensor = _store_val_in_tensor(min_val)
            self._metadata_cache["min_seqlen"] = min_seqlen_tensor
        return _load_val_from_tensor(min_seqlen_tensor)

    # Private accessors used for treating min / max seqlen as inner tensors for
    # flatten / unflatten. These must be properties to work with the traceable wrapper
    # subclass logic. These do not compute / cache if not present.
    @property
    def _max_seqlen_tensor(self) -> Optional[torch.Tensor]:
        return self._metadata_cache.get("max_seqlen", None)

    @_max_seqlen_tensor.setter
    def _max_seqlen_tensor(self, val: Optional[torch.Tensor]) -> None:
        self._metadata_cache["max_seqlen"] = val

    @property
    def _min_seqlen_tensor(self) -> Optional[torch.Tensor]:
        return self._metadata_cache.get("min_seqlen", None)

    @_min_seqlen_tensor.setter
    def _min_seqlen_tensor(self, val: Optional[torch.Tensor]) -> None:
        self._metadata_cache["min_seqlen"] = val

    # These are old private @property accessors that are kept around for internal BC
    # reasons. TODO: Remove these!
    @property
    def _max_seqlen(self):
        return self._get_max_seqlen()

    @property
    def _min_seqlen(self):
        return self._get_min_seqlen()

    # Convenience accessors that return a min / max seqlen if one is present and do NOT
    # compute / cache them if they're not.
    @property
    def _maybe_max_seqlen(self) -> Optional[int]:
        mt = self._max_seqlen_tensor
        return None if mt is None else _load_val_from_tensor(mt)

    @property
    def _maybe_min_seqlen(self) -> Optional[int]:
        mt = self._min_seqlen_tensor
        return None if mt is None else _load_val_from_tensor(mt)

    def __repr__(self):  # type: ignore[override]
        # We should implement this in torch/_tensor_str.py instead
        grad_fn_str = (
            f", requires_grad={self.requires_grad}" if self.requires_grad else ""
        )
        if self.grad_fn:
            grad_fn_str = f", grad_fn={self.grad_fn}"
        return f"NestedTensor(size={self._size}, offsets={self._offsets}{grad_fn_str}, contiguous={self.is_contiguous()})"

    # TODO: Remove this in favor of the default tensor subclass serialization logic.
    # We don't do this today because of https://github.com/pytorch/pytorch/issues/125622.
    def __reduce_ex__(self, proto):
        state = torch._utils._get_obj_state(self)

        # Cached PyCapsules for sizes / strides are not serializable.
        # See Note [Tensor Subclass custom size/stride caching strategy]
        self._clear_non_serializable_cached_data()
        # SymNodes are not serializable
        assert "_size" in state and "_strides" in state
        state = dict(state)
        del state["_size"]
        del state["_strides"]

        func = _rebuild_njt
        constructor_kwargs = {
            "values": self._values,
            "offsets": self._offsets,
            "lengths": self._lengths,
            "_ragged_idx": self._ragged_idx,
            "_metadata_cache": self._metadata_cache,
            "requires_grad": self.requires_grad,
        }
        args = (constructor_kwargs,)
        return (torch._tensor._rebuild_from_type_v2, (func, type(self), args, state))

    def __tensor_flatten__(self):
        ctx = {
            "requires_grad": self.requires_grad,
            "ragged_idx": self._ragged_idx,
        }
        inner_tensors = ["_values", "_offsets"]
        if self._lengths is not None:
            inner_tensors.append("_lengths")
        if self._min_seqlen_tensor is not None:
            inner_tensors.append("_min_seqlen_tensor")
        if self._max_seqlen_tensor is not None:
            inner_tensors.append("_max_seqlen_tensor")
        return inner_tensors, ctx

    @staticmethod
    def __tensor_unflatten__(inner_tensors: Dict, meta, outer_size, outer_stride):
        from torch._subclasses.fake_tensor import FakeTensor

        # inner tensors: _values, _offsets, [_lengths], [_min_seqlen], [_max_seqlen]
        assert len(inner_tensors) >= 2 and len(inner_tensors) <= 5
        values = inner_tensors["_values"]
        offsets = inner_tensors["_offsets"]
        lengths = inner_tensors.get("_lengths", None)
        min_seqlen_tensor = inner_tensors.get("_min_seqlen_tensor", None)
        max_seqlen_tensor = inner_tensors.get("_max_seqlen_tensor", None)

        metadata_cache = {}
        if min_seqlen_tensor is not None:
            metadata_cache["min_seqlen"] = min_seqlen_tensor
        if max_seqlen_tensor is not None:
            metadata_cache["max_seqlen"] = max_seqlen_tensor
        ragged_idx = meta["ragged_idx"]

        # Alternatively, we could make it the caller's responsibility to
        # cache it. But this heuristic seems simple enough.
        ragged_source = offsets if lengths is None else lengths
        if isinstance(ragged_source, FakeTensor):
            ragged_size = outer_size[ragged_idx]
            ragged_source.nested_int_memo = ragged_size

        return NestedTensor(
            values,
            offsets=offsets,
            lengths=lengths,
            requires_grad=meta["requires_grad"],
            _ragged_idx=ragged_idx,
            _metadata_cache=metadata_cache,
        )

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):  # type: ignore[override]
        # If you're wondering why there's a nested tensor with one of its
        # size = -1, see note: [NJT outer_size in AOTDispatcher]
        kwargs = {} if kwargs is None else kwargs

        # Lazy import to avoid circular dependency
        from .ops import lookup_jagged

        fn = lookup_jagged(func, *args, **kwargs)
        if fn is not None:
            return fn(*args, **kwargs)

        # Poor man's redispatch for composite ops. This becomes relevant under inference
        # mode, where disabling autograd key dispatch prevents decomposition.
        all_dks = (
            # We want to handle both the cases where NestedTensor overrides the
            # composite implicit autograd kernel, and the case where it doesn't.
            # Prioritize calling into NestedTensor's kernel if it exists.
            torch._C.DispatchKey.CompositeImplicitAutogradNestedTensor,
            torch._C.DispatchKey.CompositeImplicitAutograd,
        )
        for dk in all_dks:
            if torch._C._dispatch_has_kernel_for_dispatch_key(func.name(), dk):
                with torch.overrides.enable_reentrant_dispatch():
                    return func._op_dk(dk, *args, **kwargs)

        raise NotImplementedError(func)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        from torch.fx.experimental.proxy_tensor import maybe_enable_thunkify

        from .ops import jagged_torch_function

        # This should be removed after
        # https://github.com/pytorch/pytorch/pull/125941/ lands
        with maybe_enable_thunkify():
            try:
                return jagged_torch_function(func, *args, **kwargs)
            except NotImplementedError:
                pass
            with torch._C.DisableTorchFunctionSubclass():
                return func(*args, **kwargs)


# NB: These fake view autograd.Functions are superseded by real view ops. Don't use them!
# TODO: Remove ViewBufferFromNested, ViewNestedFromBuffer, and buffer_from_jagged once the
# internal BC period has passed.


# Not actually a view!
class ViewBufferFromNested(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: NestedTensor):  # type: ignore[override]
        ctx.save_for_backward(x.offsets())
        ctx.metadata_cache = x._metadata_cache
        ctx.ragged_idx = x._ragged_idx
        return x._values

    @staticmethod
    def backward(ctx, gO: torch.Tensor):  # type: ignore[override]
        (offsets,) = ctx.saved_tensors
        return NestedTensor(
            gO,
            offsets=offsets,
            _metadata_cache=ctx.metadata_cache,
            _ragged_idx=ctx.ragged_idx,
        )


# Not actually a view!
class ViewNestedFromBuffer(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        values: torch.Tensor,
        offsets: torch.Tensor,
        metadata_cache: Optional[Dict[str, Any]] = None,
    ):  # type: ignore[override]
        # maintain BC with this usages of this where the seqlens are stuffed
        # directly into the metadata cache as non-Tensors / ints
        if metadata_cache is not None:
            min_seqlen = metadata_cache.get("min_seqlen", None)
            max_seqlen = metadata_cache.get("max_seqlen", None)
            if min_seqlen is not None and not isinstance(min_seqlen, torch.Tensor):
                metadata_cache["min_seqlen"] = _store_val_in_tensor(min_seqlen)
            if max_seqlen is not None and not isinstance(max_seqlen, torch.Tensor):
                metadata_cache["max_seqlen"] = _store_val_in_tensor(max_seqlen)
        return NestedTensor(
            values.detach(),
            offsets=offsets,
            _metadata_cache=metadata_cache,
        )

    @staticmethod
    def backward(ctx, gO: NestedTensor):  # type: ignore[override]
        return gO._values, None, None


def buffer_from_jagged(jagged):
    return ViewBufferFromNested.apply(jagged)


# Need to make it obvious that users should be passing in offsets
def jagged_from_list(
    tensors: List[torch.Tensor],
    offsets: Optional[torch.Tensor],
    dtype=None,
    device=None,
) -> tuple[NestedTensor, torch.Tensor]:
    """Constructs a NestedTensor backed by jagged layout from a list of tensors"""

    if len(tensors) == 0:
        raise RuntimeError("Cannot construct a nested tensor from an empty tensor list")
    if not len(set(t.dtype for t in tensors)) == 1:  # noqa: C401
        raise RuntimeError(
            "When constructing a nested tensor, all tensors in list must have the same dtype"
        )
    if not len(set(t.device for t in tensors)) == 1:  # noqa: C401
        raise RuntimeError(
            "When constructing a nested tensor, all tensors in list must be on the same device"
        )
    if not len(set(t.dim() for t in tensors)) == 1:  # noqa: C401
        raise RuntimeError(
            "When constructing a nested tensor, all tensors in list must have the same dim"
        )
    component_dim = tensors[0].dim()
    if component_dim == 0:
        raise RuntimeError(
            "Cannot construct a nested tensor from a list of zero-dim tensors"
        )

    # Check that the NT is representable by the jagged layout, which
    # allows for a single ragged dimension after the batch dim.
    # e.g. (B, *, D_0, ..., D_N), (B, D_0, *, ..., D_N), etc.
    sizes = [t.shape for t in tensors]
    ragged_idx = None
    for d in range(component_dim):
        dim_is_ragged = any(size[d] != sizes[0][d] for size in sizes)
        if dim_is_ragged:
            if ragged_idx is None:
                # add 1 to convert to outer NJT dim space
                ragged_idx = d + 1
            else:
                raise RuntimeError(
                    "Cannot represent given tensor list as a nested tensor with the jagged layout. "
                    "Note that the jagged layout only allows for a single ragged dimension. "
                    "For example: (B, *, D_0, D_1, ..., D_N), with ragged * dim."
                )

    # allow for a rectangular NJT and default the ragged dim next to the batch dim
    if ragged_idx is None:
        ragged_idx = 1

    # Set properties appropriately.
    values = torch.cat(tensors, dim=(ragged_idx - 1))
    to_kwargs = {}
    if device is not None:
        to_kwargs["device"] = device
    if dtype is not None:
        to_kwargs["dtype"] = dtype
    values = values.to(**to_kwargs)

    # Calculate jagged offsets if not provided.
    if offsets is None:
        # Jagged layout specifies that offsets are stored as int64 on the same device as values.
        # TODO: An alternative way to construct offsets is to use F.pad. This avoids creating
        # an extra leaf tensor during the forward, potentially resolving compatibility issues.
        offsets = torch.cat(
            [
                torch.zeros(1, dtype=torch.int64, device=values.device),
                torch.tensor(
                    [s[ragged_idx - 1] for s in sizes], device=values.device
                ).cumsum(dim=0),
            ]
        )

    # compute this now since it's easy
    min_seqlen = min(t.shape[ragged_idx - 1] for t in tensors)
    max_seqlen = max(t.shape[ragged_idx - 1] for t in tensors)
    ret_nt = nested_view_from_values_offsets(
        values,
        offsets,
        min_seqlen=min_seqlen,
        max_seqlen=max_seqlen,
        ragged_idx=ragged_idx,
    )
    return (ret_nt, offsets)  # type: ignore[return-value]


def jagged_from_tensor_and_lengths(
    tensor: torch.Tensor, starts: torch.Tensor, lengths: torch.Tensor
) -> tuple[NestedTensor, torch.Tensor, Optional[torch.Tensor]]:
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
            "your start and length must be Tensors that broadcast to input.shape[0]"
        )

    # Calculate jagged offsets
    assert len(tensor.shape) >= 2, (
        "tensor must at least be 2D for the nested narrow op to work"
    )
    max_seq_len = tensor.shape[1]
    offset_lengths = max_seq_len * torch.arange(
        0, batch_size, dtype=torch.int64, device=tensor.device
    )
    # Jagged layout specifies that offsets are stored as int64 on the same device as values.
    offsets = torch.cat(
        [
            start_list + offset_lengths,
            (start_list[-1] + offset_lengths[-1] + length_list[-1]).unsqueeze(0),
        ]
    )

    # Reshape buffer to flatten the 1st and 2nd dimension (view used to enforce non-copy)
    if len(tensor.shape) > 2:
        values = tensor.view(-1, *tensor.shape[2:])
    else:
        values = tensor.view(-1)

    # Check if offsets and lengths make it possibly contiguous and return a regular NT
    is_contiguous = True
    orig_dim = tensor.shape[1]
    if torch.any(length_list[1:-1].ne(orig_dim)):
        is_contiguous = False
    if torch.any(offsets[1:-2].diff().ne(orig_dim)):
        is_contiguous = False
    if offsets[0] + length_list[0] != orig_dim:
        is_contiguous = False

    actual_max_seqlen = int(torch.max(lengths).item())
    min_seqlen = int(torch.min(lengths).item())

    if is_contiguous:
        ret_nt = nested_view_from_values_offsets(
            values[offsets[0] : offsets[-1]],
            offsets - offsets[0],
            min_seqlen=min_seqlen,
            max_seqlen=actual_max_seqlen,
        )
    else:
        ret_nt = nested_view_from_values_offsets_lengths(
            values,
            offsets,
            length_list,
            min_seqlen=min_seqlen,
            max_seqlen=actual_max_seqlen,
        )

    return (ret_nt, offsets, None if is_contiguous else length_list)


# NB: A dummy arg is required so that NestedTensor.__torch_dispatch__() is invoked
# for _nested_view_from_values_offsets(). Sizes don't matter much, but they shouldn't be
# 0/1 because the dummy can be fake-ified and we want to avoid specializing.
# This arg is otherwise unused.
_dummy_instance: Optional[torch.Tensor] = None


def _nt_view_dummy() -> torch.Tensor:
    global _dummy_instance
    if _dummy_instance is None:
        _dummy_instance = NestedTensor(
            values=torch.zeros(3, 3, device="meta"),
            offsets=torch.zeros(3, device="meta", dtype=torch.int64),
        ).detach()
    return _dummy_instance


def nested_view_from_values_offsets(
    values, offsets, ragged_idx=1, min_seqlen=None, max_seqlen=None
):
    min_seqlen_tensor = None
    if min_seqlen is not None:
        min_seqlen_tensor = _store_val_in_tensor(min_seqlen)

    max_seqlen_tensor = None
    if max_seqlen is not None:
        max_seqlen_tensor = _store_val_in_tensor(max_seqlen)

    return torch._nested_view_from_jagged(  # type: ignore[attr-defined]
        values,
        offsets,
        _nt_view_dummy(),
        None,
        ragged_idx,
        min_seqlen_tensor,
        max_seqlen_tensor,
    )  # type: ignore[return-value]


def nested_view_from_values_offsets_lengths(
    values, offsets, lengths, ragged_idx=1, min_seqlen=None, max_seqlen=None
):
    min_seqlen_tensor = None
    if min_seqlen is not None:
        min_seqlen_tensor = _store_val_in_tensor(min_seqlen)

    max_seqlen_tensor = None
    if max_seqlen is not None:
        max_seqlen_tensor = _store_val_in_tensor(max_seqlen)

    return torch._nested_view_from_jagged(  # type: ignore[attr-defined]
        values,
        offsets,
        _nt_view_dummy(),
        lengths,
        ragged_idx,
        min_seqlen_tensor,
        max_seqlen_tensor,
    )  # type: ignore[return-value]


def nested_from_padded(
    padded, offsets, ragged_idx=1, min_seqlen=None, max_seqlen=None, sum_S=None
):
    min_seqlen_tensor = None
    if min_seqlen is not None:
        min_seqlen_tensor = _store_val_in_tensor(min_seqlen)

    max_seqlen_tensor = None
    if max_seqlen is not None:
        max_seqlen_tensor = _store_val_in_tensor(max_seqlen)

    return torch._nested_from_padded_tensor(
        padded,
        offsets,
        _nt_view_dummy(),
        ragged_idx,
        min_seqlen_tensor,
        max_seqlen_tensor,
        sum_S,
    )
