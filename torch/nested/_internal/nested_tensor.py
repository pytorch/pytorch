# mypy: allow-untyped-defs
from typing import *  # noqa: F403
from collections import namedtuple
from typing import Tuple

import torch
from torch._C import DispatchKey, DispatchKeySet
from torch._prims_common import is_expandable_to
from torch.nested._internal.cached_tensor import CachedTensor
from torch.nested._internal.nested_int import NestedIntNode, get_metadata
from torch.nested._internal.offload_tensor import (
    init_offload_tensor_registry,
    OffloadTensorRegistry,
)
from torch.nested._internal.tensor_registry import register_tensor, try_get_int
from torch.nested._internal.utils import _try_get_fake_mode


def update_tensor_registry_hook(host_tensor, device_tensor):
    register_tensor(host_tensor, try_get_int(device_tensor))


init_offload_tensor_registry(OffloadTensorRegistry(update_tensor_registry_hook))


# Fully represents raggedness
source_fields = ("_device_lengths", "_device_offsets", "_host_lengths", "_host_offsets")
# Derived fields
extra_fields = (
    "_max_seqlen_tensor",
    "_min_seqlen_tensor",
    "_inverse_indices",
)
# UnpackResult = namedtuple("UnpackResult", source_fields + extra_fields)


def get_tensor_symint(metadata_tensor, *, coeff=1):
    if fake_mode := _try_get_fake_mode(metadata_tensor):
        return fake_mode.get_nested_int(cache=metadata_tensor, coeff=coeff)

    return torch.SymInt(NestedIntNode(metadata_tensor, coeff))


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
    _non_contig_offsets: torch.Tensor
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
        values: torch.Tensor,
        metadata: torch.Tensor,  # This will be a subclass representing raggedness
        non_contig_offsets: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        ks = DispatchKeySet(DispatchKey.NestedTensor)
        ks = ks.add(DispatchKey.AutogradNestedTensor)

        # Only support jagged for now.
        assert metadata is not None
        assert metadata.ndim == 1
        assert not isinstance(values, NestedTensor)
        assert values.device == metadata.device

        # Query cache for the symint associated with offsets or lengths
        # (create a new one if needed).
        ragged_size = get_tensor_symint(metadata, coeff=1)
        _ragged_idx = kwargs.get("_ragged_idx", 1)

        # When non_contig_offsets is not None, metadata represents lengths
        B = metadata.shape[0] - 1 if non_contig_offsets is None else metadata.shape[0]

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

    def __init__(self, values, metadata, non_contig_offsets=None, **kwargs):
        super().__init__()

        self._values = values
        self._non_contig_offsets = non_contig_offsets

        # collapsed ragged dim must always be dynamic
        torch._dynamo.maybe_mark_dynamic(self, self._ragged_idx)
        torch._dynamo.maybe_mark_dynamic(self._values, self._ragged_idx - 1)

    def values(self):
        # dispatch to get proper view relationship
        return torch._nested_get_values(self)  # type: ignore[attr-defined]

    # Public APIs for lengths/offsets returns plain tensor for BC reasons, e.g. user
    # can access .data_ptr() on it, etc.
    # .lengths() and .offsets() do not automatically do conversion between lengths
    # and offsets. If the requested variant does not exist, then None is returned.
    def offsets(self) -> Optional[torch.Tensor]:
        if self._non_contig_offsets is not None:
            return self._non_contig_offsets
        if self.is_cpu:
            return self._host_offsets
        else:
            return self._device_offsets.restore() if self._device_offsets is not None else None

    def lengths(self) -> Optional[torch.Tensor]:
        if self.is_cpu:
            return self._host_lengths
        else:
            return self._device_lengths.restore() if self._device_lengths is not None else None

    def __getattr__(self, name) -> Optional[torch.Tensor]:
        if name not in source_fields + extra_fields + ("_lengths", "_offsets"):
            raise AttributeError(
                f"{type(self).__name__} object has no attribute '{name}'"
            )

        metadata = get_metadata(self.shape[self._ragged_idx])

        if name == "_offsets":
            return self.offsets()
            
        elif name == "_lengths":
            return self.lengths()
            
        return metadata.metadata.get(name)

    @property
    def _metadata(self):
        from torch.nested._internal.nested_int import NestedIntNode

        node = self.shape[self._ragged_idx].node
        if isinstance(node, NestedIntNode):
            metadata = node.nested_int_cache()
        else:
            metadata = node.hint.node.nested_int_cache()
        return metadata

    # Allow user to access nt._metadata_cache for BC
    # This is dangerous to allow mutation!
    @property
    def _metadata_cache(self):
        ret = self._metadata.metadata.copy()
        if (max_seqlen := ret.get("_max_seqlen_tensor")) is not None:
            del ret["_max_seqlen_tensor"]
            ret["max_seqlen"] = max_seqlen
        if (min_seqlen := ret.get("_min_seqlen_tensor")) is not None:
            del ret["_min_seqlen_tensor"]
            ret["min_seqlen"] = min_seqlen
        return ret

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
        mt = self._metadata.metadata.get("_max_seqlen_tensor")
        return None if mt is None else _load_val_from_tensor(mt)

    @property
    def _maybe_min_seqlen(self) -> Optional[int]:
        mt = self._metadata.metadata.get("_min_seqlen_tensor")
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
            "metadata": self._metadata,
            "non_contig_offsets": self._non_contig_offsets,
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
        inner_tensors = ["_values", "_metadata"]
        if self._non_contig_offsets is not None:
            inner_tensors.append("_non_contig_offsets")
        return inner_tensors, ctx

    @staticmethod
    def __tensor_unflatten__(inner_tensors: Dict, meta, outer_size, outer_stride):
        return NestedTensor(
            inner_tensors["_values"],
            inner_tensors["_metadata"],
            non_contig_offsets=inner_tensors.get("_non_contig_offsets"),
            requires_grad=meta["requires_grad"],
            _ragged_idx=meta["ragged_idx"],
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
        ctx.save_for_backward(x.offsets())
        ctx.metadata_cache = x._metadata_cache
        ctx.ragged_idx = x._ragged_idx
        return x._values

    @staticmethod
    def backward(ctx, gO: torch.Tensor):  # type: ignore[override]
        (offsets,) = ctx.saved_tensors
        return _construct_nested_tensor_compat(
            values=gO,
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
        # This maintains BC with this usages of this where the seqlens are stuffed
        # directly into the metadata cache as non-Tensors / ints
        return _construct_nested_tensor_compat(
            values=values.detach(),
            offsets=offsets,
            lengths=None,
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
) -> Tuple[NestedTensor, torch.Tensor]:
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


def normalize_lengths_offsets(metadata):
    from torch.nested._internal.wrappers import make_offload_tensor
    # Does the follow conversion:
    # {
    #     "lengths": torch.Tensor,
    #     "offsets": torch.Tensor,
    #     ...
    # }
    # =>
    # {
    #     "_host_lengths": torch.Tensor,
    #     "_host_offsets": torch.Tensor,
    #     "_device_lengths": OffloadTensor(),
    #     "_device_offsets": OffloadTensor(),
    #     ...
    # }
    # Need a layer processing the wrapping process
    if (offsets := metadata.get("offsets")) is not None:
        del metadata["offsets"]
        if offsets.is_cpu:
            metadata["_host_offsets"] = offsets
        else:
            metadata["_device_offsets"] = make_offload_tensor(device_tensor=offsets)

    if (lengths := metadata.get("lengths")) is not None:
        del metadata["lengths"]
        if lengths.is_cpu:
            metadata["_host_lengths"] = lengths
        else:
            metadata["_device_lengths"] = make_offload_tensor(device_tensor=lengths)


def normalize_min_max_seqlen(metadata):
    # Does the follow conversion:
    # {
    #     "max_seqlen": Union[torch.Tensor, int]
    #     "min_seqlen": Union[torch.Tensor, int]
    #     ...
    # }
    # =>
    # {
    #     "_max_seqlen_tensor": torch.Tensor,
    #     "_min_seqlen_tensor": torch.Tensor,
    #     ...
    # }
    if (max_seqlen := metadata.get("max_seqlen")) is not None:
        del metadata["max_seqlen"]
        if isinstance(max_seqlen, (int, torch.SymInt)):
            metadata["_max_seqlen_tensor"] = _store_val_in_tensor(max_seqlen)
        else:
            metadata["_max_seqlen_tensor"] = max_seqlen
    if metadata.get("_max_seqlen_tensor") is not None:
        torch._dynamo.mark_dynamic(metadata["_max_seqlen_tensor"], 0)
    if (min_seqlen := metadata.get("min_seqlen")) is not None:
        del metadata["min_seqlen"]
        if isinstance(min_seqlen, (int, torch.SymInt)):
            metadata["_min_seqlen_tensor"] = _store_val_in_tensor(min_seqlen)
        else:
            metadata["_min_seqlen_tensor"] = min_seqlen
    if metadata.get("_min_seqlen_tensor") is not None:
        torch._dynamo.mark_dynamic(metadata["_min_seqlen_tensor"], 0)


def _make_ragged_tensors_compat(*, offsets, lengths, min_seqlen, max_seqlen):
    from torch.nested._internal.wrappers import make_cached_tensor

    assert offsets is not None or lengths is not None
    metadata = {}
    if offsets is not None and lengths is not None:
        non_contig_offsets = offsets
        if isinstance(lengths, CachedTensor):
            if lengths.is_cpu:
                metadata["_host_lengths"] = lengths._host_lengths
            else:
                metadata["_device_lengths"] = lengths._device_lengths
        else:
            metadata["lengths"] = lengths
    else:
        non_contig_offsets = None
        if isinstance(offsets, CachedTensor):
            if offsets.is_cpu:
                metadata["_host_offsets"] = offsets._host_offsets
            else:
                metadata["_device_offsets"] = offsets._device_offsets
        else:
            metadata["offsets"] = offsets
    if min_seqlen is not None:
        metadata["min_seqlen"] = min_seqlen
    if max_seqlen is not None:
        metadata["max_seqlen"] = max_seqlen
    normalize_lengths_offsets(metadata)
    normalize_min_max_seqlen(metadata)
    metadata_tensor = make_cached_tensor(metadata)
    return metadata_tensor, non_contig_offsets


# Useful for tests that previously constructed the NestedTensor directly via
# the raw constructor.
# kwargs that are not _metadata_cache or _ragged_idx are ignored.
# Fields in _metadata_cache that are not _metadata_cache are ignored.
def _construct_nested_tensor_compat(
    values: torch.Tensor,
    offsets: torch.Tensor,
    lengths: Optional[torch.Tensor] = None,
    **kwargs,
):
    ragged_idx = kwargs.get("_ragged_idx", 1)
    metadata_cache = kwargs.get("_metadata_cache") or {}

    max_seqlen = metadata_cache.get("max_seqlen")
    min_seqlen = metadata_cache.get("min_seqlen")

    metadata, non_contig_offsets = _make_ragged_tensors_compat(
        offsets=offsets,
        lengths=lengths,
        max_seqlen=max_seqlen,
        min_seqlen=min_seqlen,
    )
    return NestedTensor(
        values=values,
        metadata=metadata,
        non_contig_offsets=non_contig_offsets,
        _ragged_idx=ragged_idx,
    )


def nested_view_from_values_offsets(
    values, offsets, ragged_idx=1, min_seqlen=None, max_seqlen=None
):
    metadata_tensor, non_contig_offsets = _make_ragged_tensors_compat(
        offsets=offsets,
        lengths=None,
        max_seqlen=max_seqlen,
        min_seqlen=min_seqlen,
    )
    assert non_contig_offsets is None
    ret = torch._nested_view_from_jagged(  # type: ignore[attr-defined]
        values,
        metadata_tensor,
        None,
        ragged_idx,
    )  # type: ignore[return-value]
    return ret


def nested_view_from_values_offsets_lengths(
    values, offsets, lengths, ragged_idx=1, min_seqlen=None, max_seqlen=None
):
    metadata_tensor, non_contig_offsets = _make_ragged_tensors_compat(
        offsets=offsets,
        lengths=lengths,
        max_seqlen=max_seqlen,
        min_seqlen=min_seqlen,
    )
    return torch._nested_view_from_jagged(  # type: ignore[attr-defined]
        values,
        metadata_tensor,
        non_contig_offsets,
        ragged_idx,
    )  # type: ignore[return-value]


def nested_from_padded(
    padded, offsets, ragged_idx=1, min_seqlen=None, max_seqlen=None, sum_S=None
):
    metadata_tensor, non_contig_offsets = _make_ragged_tensors_compat(
        offsets=offsets,
        lengths=None,
        max_seqlen=max_seqlen,
        min_seqlen=min_seqlen,
    )
    assert non_contig_offsets is None
    return torch._nested_from_padded_tensor(
        padded,
        metadata_tensor,
        ragged_idx,
        sum_S,
    )
