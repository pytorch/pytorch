# mypy: allow-untyped-defs
from typing import *  # noqa: F403
import functools
from typing import Tuple

import torch
from torch._C import DispatchKey, DispatchKeySet
from torch._prims_common import is_expandable_to

from torch.nested._internal.metadata_cache import (
    add_entry,
    merge_caches,
    register_tensor,
    TreeCache,
    try_get_cache,
)
from torch.nested._internal.nested_int import get_nested_symint


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


# The order of these keys determines the priority for merging
_METADATA_KEYS = [
    "lengths",
    "offsets",
    "max_seqlen_tensor",
    "min_seqlen_tensor",
    "dummy_entry",
]


def get_device_and_host_metadata(**kwargs):
    device_meta, host_meta = {}, {}
    for key in _METADATA_KEYS:
        if key in kwargs and kwargs[key] is not None:
            meta = host_meta if kwargs[key].is_cpu else device_meta
            meta[key] = kwargs[key]
    return device_meta, host_meta


def get_nested_cache(host_meta, device_meta) -> TreeCache:
    # Collect existing caches
    caches = []
    for k in _METADATA_KEYS:
        if k in host_meta:
            _cache = try_get_cache(host_meta[k])
            if _cache is not None:
                caches.append(_cache)
        if k in device_meta:
            _cache = try_get_cache(device_meta[k])
            if _cache is not None:
                caches.append(_cache)
    # Merge them
    cache = None
    if len(caches) > 0:
        # print("existing caches found!", len(caches))
        cache = functools.reduce(merge_caches, caches)
        # print("found: ", cache, [id(v) for v in host_meta.values()])
    # else:
    #     print("no cache found with: ", [id(v) for v in host_meta.values()])
    #     breakpoint()

    # Add new entries
    for k, v in host_meta.items():
        if cache is None or cache.data.get(k) is None:
            # Creates a new cache if None if cache=None
            cache = add_entry(cache, k, v)

    return cache


class NestedTensor(torch.Tensor):
    _values: torch.Tensor  # type: ignore[assignment]
    _device_metadata: Dict[str, Any]
    _host_metadata: Dict[str, Any]
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
    _size: Tuple[int, ...]
    _strides: Tuple[int, ...]
    # Indicates that the nth dimension is ragged
    _ragged_idx: int

    @staticmethod
    def __new__(
        cls,
        values,
        device_meta,
        host_meta,
        **kwargs,
    ):
        ks = DispatchKeySet(DispatchKey.NestedTensor)
        ks = ks.add(DispatchKey.AutogradNestedTensor)

        offsets = (
            device_meta.get("offsets")
            if device_meta.get("offsets") is not None
            else host_meta.get("offsets")
        )
        lengths = (
            device_meta.get("lengths")
            if device_meta.get("lengths") is not None
            else host_meta.get("lengths")
        )

        # Only support jagged for now.
        # TODO(soulitzer): we don't actually run these checks on both device/host
        # if both are provided.
        assert offsets is not None
        assert offsets.ndim == 1
        assert not isinstance(values, NestedTensor)
        assert values.device == offsets.device

        print(
            "construct with",
            [id(v) for v in host_meta.values()],
            [id(v) for v in device_meta.values()],
        )

        # Always have something in the host_meta.
        assert "dummy_entry" in host_meta

        cache = get_nested_cache(host_meta, device_meta)

        for v in device_meta.values():
            register_tensor(cache, v)

        ragged_size = get_nested_symint(cache, coeff=1)

        _ragged_idx = kwargs.get("_ragged_idx", 1)
        B = offsets.shape[0] - 1
        if lengths is not None:
            assert B == lengths.shape[0]

        # subtract 1 to convert to values dim space
        r = _ragged_idx - 1
        _size = (B, *values.shape[:r], ragged_size, *values.shape[r + 1 :])
        stride = values.stride()
        _strides = (ragged_size * stride[r], *stride)

        r = torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
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

    def __init__(self, values, device_meta, host_meta, **kwargs):
        super().__init__()

        self._values = values
        self._host_meta = host_meta
        self._device_meta = device_meta

        # collapsed ragged dim must always be dynamic
        torch._dynamo.maybe_mark_dynamic(self, self._ragged_idx)
        torch._dynamo.maybe_mark_dynamic(self._values, self._ragged_idx - 1)

        # min / max sequence length should be dynamic if present
        for meta in (host_meta, device_meta):
            for key in ("max_seqlen_tensor", "min_seqlen_tensor"):
                if meta.get(key) is not None:
                    print("marking as dynamic: ", key)
                    torch._dynamo.mark_dynamic(meta[key], 0)

    # Private accessors used for treating min / max seqlen as inner tensors for
    # flatten / unflatten. These must be properties to work with the traceable wrapper
    # subclass logic. These do not compute / cache if not present.
    @property
    def _device_offsets(self):
        return self._device_meta.get("offsets")

    @property
    def _host_offsets(self):
        return self._host_meta.get("offsets")

    @property
    def _device_lengths(self):
        return self._device_meta.get("lengths")

    @property
    def _host_lengths(self):
        return self._host_meta.get("lengths")

    @property
    def _device_max_seqlen_tensor(self):
        return self._device_meta.get("max_seqlen_tensor")

    @property
    def _host_max_seqlen_tensor(self):
        return self._host_meta.get("max_seqlen_tensor")

    @property
    def _device_min_seqlen_tensor(self):
        return self._device_meta.get("min_seqlen_tensor")

    @property
    def _host_min_seqlen_tensor(self):
        return self._host_meta.get("min_seqlen_tensor")

    @property
    def _host_dummy_entry(self):
        return self._host_meta.get("dummy_entry")

    # Wrappers on top of the private accessors to abstract over device / host
    # For these APIs, device metadata is preferred if present.
    # What happens in the cuda case vs cpu case.
    # For the cuda case: values devices is cuda
    # The NJT always at least has the offsets on cuda, it may also have the host offsets
    # cached, but not always, this means when we return _offsets

    # Ban cpu NJT from caching device offsets for simplicitly
    @property
    def _offsets(self):
        return (
            self._device_offsets
            if self._device_offsets is not None
            else self._host_offsets
        )

    @property
    def _lengths(self):
        return (
            self._device_lengths
            if self._device_lengths is not None
            else self._host_lengths
        )

    @property
    def _max_seqlen_tensor(self) -> Optional[torch.Tensor]:
        return (
            self._device_max_seqlen_tensor
            if self._device_max_seqlen_tensor is not None
            else self._host_max_seqlen_tensor
        )

    @property
    def _min_seqlen_tensor(self) -> Optional[torch.Tensor]:
        return (
            self._device_min_seqlen_tensor
            if self._device_min_seqlen_tensor is not None
            else self._host_min_seqlen_tensor
        )

    def values(self):
        # dispatch to get proper view relationship
        return torch._nested_get_values(self)  # type: ignore[attr-defined]

    def offsets(self):
        return self._offsets

    def lengths(self):
        return self._lengths

    # Helper for _get_{max,min}_seqlen
    def _compute_and_store_max_min_seqlen(self, func, name):
        val = _get_sdpa_extreme_seqlen(
            func,
            self._offsets.diff() if self._lengths is None else self._lengths,
        )
        val_tensor = _store_val_in_tensor(val)
        meta = (
            self._device_meta if self._device_offsets is not None else self._host_meta
        )
        meta[name] = val_tensor
        return val_tensor

    # Private accessor functions for min / max sequence length. They're
    # purposefully not @properties because those don't work with PT2 (yet).
    # These compute / cache if not present.
    # TODO: Revisit this when @properties are better supported by PT2. I think the ideal
    # state would be to have public @properties for min / max sequence length that compile
    # (including setters).
    def _get_max_seqlen(self):
        # Prefer device metadata if present
        max_seqlen_tensor = self._max_seqlen_tensor
        if max_seqlen_tensor is None:
            max_seqlen_tensor = self._compute_and_store_max_min_seqlen(
                torch.max, "max_seqlen_tensor"
            )
        return _load_val_from_tensor(max_seqlen_tensor)

    def _get_min_seqlen(self):
        min_seqlen_tensor = self._min_seqlen_tensor
        if min_seqlen_tensor is None:
            min_seqlen_tensor = self._compute_and_store_max_min_seqlen(
                torch.min, "min_seqlen_tensor"
            )
        return _load_val_from_tensor(min_seqlen_tensor)

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
        return f"NestedTensor(size={self._size}, offsets={self._offsets}{grad_fn_str}, contiguous={self._lengths is None})"

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
            "_host_meta": self._host_meta,
            "_device_meta": self._device_meta,
            "_ragged_idx": self._ragged_idx,
            "requires_grad": self.requires_grad,
        }
        args = (constructor_kwargs,)
        return (torch._tensor._rebuild_from_type_v2, (func, type(self), args, state))

    def __tensor_flatten__(self):
        ctx = {
            "requires_grad": self.requires_grad,
            "ragged_idx": self._ragged_idx,
        }
        inner_tensors = ["_values"]
        for x_name, x_meta in (
            ("device", self._device_meta),
            ("host", self._host_meta),
        ):
            for k in _METADATA_KEYS:
                if k in x_meta:
                    inner_tensors.append(f"_{x_name}_{k}")
        return inner_tensors, ctx

    @staticmethod
    def __tensor_unflatten__(inner_tensors: Dict, ctx, outer_size, outer_stride):
        # inner tensors: _values, _offsets, [_lengths], [_min_seqlen], [_max_seqlen]
        assert len(inner_tensors) >= 2 and len(inner_tensors) <= 5

        print("__tensor_unflatten__: ", inner_tensors)

        device_meta, host_meta = {}, {}
        for suffix in _METADATA_KEYS:
            if (device_k := "_device_" + suffix) in inner_tensors:
                device_meta[suffix] = inner_tensors[device_k]
            if (host_k := "_host_" + suffix) in inner_tensors:
                host_meta[suffix] = inner_tensors[host_k]

        print("__tensor_unflatten__: ", device_meta, host_meta)

        return NestedTensor(
            inner_tensors["_values"],
            device_meta=device_meta,
            host_meta=host_meta,
            requires_grad=ctx["requires_grad"],
            _ragged_idx=ctx["ragged_idx"],
        )

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
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
        dk = torch._C.DispatchKey.CompositeImplicitAutogradNestedTensor
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
        # In theory tensors should be saved via ctx.save_for_backward, but
        # we'd have to annoyingly explode/rebuild the metadata.
        ctx.device_meta = x._device_meta
        ctx.host_meta = x._host_meta
        ctx.ragged_idx = x._ragged_idx
        return x._values

    @staticmethod
    def backward(ctx, gO: torch.Tensor):  # type: ignore[override]
        return NestedTensor(
            gO,
            device_meta=ctx.device_meta,
            host_meta=ctx.host_meta,
            _ragged_idx=ctx.ragged_idx,
        )


# Not actually a view!
class ViewNestedFromBuffer(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        values: torch.Tensor,
        offsets: torch.Tensor,
        # This is public API?
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

        device_meta, host_meta = {}, {}
        for prefix in ("min", "max"):
            key = f"{prefix}_seqlen"
            if key in metadata_cache:
                name, meta = (
                    ("device", device_meta)
                    if metadata_cache[key].is_cpu
                    else ("host", host_meta)
                )
                meta[f"_{name}_{key}_tensor"] = metadata_cache[key]

        return NestedTensor(
            values.detach(),
            device_meta=device_meta,
            host_meta=host_meta,
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
        # TODO: An alternative way to construct offsets is to use F.pad. This avoids creating
        # an extra leaf tensor during the forward, potentially resolving compatibility issues.
        offsets = torch.cat(
            [
                torch.zeros(1, dtype=torch.int64, device=values.device),
                torch.tensor([s[0] for s in sizes], device=values.device).cumsum(dim=0),
            ]
        )

    # compute this now since it's easy
    min_seqlen = min(t.shape[0] for t in tensors)
    max_seqlen = max(t.shape[0] for t in tensors)
    ret_nt = nested_view_from_values_offsets(
        values, offsets, min_seqlen=min_seqlen, max_seqlen=max_seqlen
    )
    return (ret_nt, offsets)  # type: ignore[return-value]


def jagged_from_tensor_and_lengths(
    tensor: torch.Tensor, starts: torch.Tensor, lengths: torch.Tensor
) -> Tuple[NestedTensor, torch.Tensor, Optional[torch.Tensor]]:
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
    assert (
        len(tensor.shape) >= 2
    ), "tensor must at least be 2D for the nested narrow op to work"
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
        device_meta = {"offsets": torch.zeros(3, device="meta", dtype=torch.int64)}
        _dummy_instance = NestedTensor(
            values=torch.zeros(3, 3, device="meta"),
            device_meta=device_meta,
            host_meta={"dummy_entry": torch.empty((0,), device="cpu")},
        ).detach()
    return _dummy_instance


def nested_view_from_values_offsets(
    values, offsets, ragged_idx=1, min_seqlen=None, max_seqlen=None
):
    # Where do we add to the cache? inside the torch dispatch right?
    # It's inside the custom op, then its fine?
    # If it is in the torch dispatch, then we have to still accept the
    # exploded metadata as argument in dynamo, which is fine?
    min_seqlen_tensor = None
    if min_seqlen is not None:
        min_seqlen_tensor = _store_val_in_tensor(min_seqlen)

    max_seqlen_tensor = None
    if max_seqlen is not None:
        max_seqlen_tensor = _store_val_in_tensor(max_seqlen)

    # Just explode doesn't seem too bad.
    return torch._nested_view_from_jagged(  # type: ignore[attr-defined]
        values,
        offsets,
        _nt_view_dummy(),
        torch.empty((0,), device="cpu"),  # dummy cache entry
        None,
        ragged_idx,
        min_seqlen_tensor,
        max_seqlen_tensor,
    )  # type: ignore[return-value]


def nested_view_from_values_offsets_lengths(
    values, offsets, lengths, ragged_idx=1, min_seqlen=None, max_seqlen=None
):
    # This is the thing where we lazily create a min_seqlen tenosr inside compile
    # in theory this shouldn't graph break.
    # The question is when did dynamo decide how to store something as a constant in
    # maybe during aot dispatch, dynamo never really fakified it.
    # we could run an experiment to test this.
    # but assuming this is true.
    # what are we doing today that triggers this?
    # but also... if dynamo does see this creation... is that okay?
    #
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
        torch.empty((0,), device="cpu"),
        lengths,
        ragged_idx,
        min_seqlen_tensor,
        max_seqlen_tensor,
    )  # type: ignore[return-value]


def nested_from_padded(
    padded, offsets, ragged_idx=1, min_seqlen=None, max_seqlen=None, sum_S=None
):
    if ragged_idx != 1:
        raise RuntimeError("nested_from_padded(): only ragged_idx=1 supported for now")

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
        torch.empty((0,), device="cpu"),
        ragged_idx,
        min_seqlen_tensor,
        max_seqlen_tensor,
        sum_S,
    )
