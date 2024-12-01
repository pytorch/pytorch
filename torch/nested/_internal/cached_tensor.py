from typing import *  # noqa: F403

import torch
from torch.utils import _pytree as pytree


def _get_source_field(metadata, source_fields):
    return next(k for k in source_fields if metadata.get(k) is not None)


def _get_source(metadata, source_fields):
    source_field = _get_source_field(metadata, source_fields)
    return metadata[source_field]


class CachedTensor(torch.Tensor):
    @staticmethod
    def __new__(
        cls,
        metadata: dict,
        source_fields=None,
        extra_fields=(),
        target_field=None,
    ):
        assert source_fields is not None
        assert any(
            metadata.get(k) is not None for k in source_fields
        ), f"CachedTensor: At least one of {source_fields} must be passed"

        # Tensor's metadata is the first non-None source field
        source = _get_source(metadata, source_fields) if target_field is None else metadata[target_field]
        shape = source.shape
        kwargs = {}
        kwargs["strides"] = source.stride()
        kwargs["storage_offset"] = source.storage_offset()
        kwargs["device"] = source.device
        kwargs["layout"] = source.layout
        kwargs["requires_grad"] = source.requires_grad
        kwargs["dtype"] = source.dtype
        out = torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)
        return out

    def __init__(self, metadata: dict, source_fields=None, extra_fields=(), target_field=None):
        self.source_fields = source_fields
        self.extra_fields = extra_fields
        self.all_fields = source_fields + extra_fields
        self.metadata = metadata

        # Compile only
        self.nested_int_ref = None

    def __repr__(self):
        source_repr = repr(_get_source(self.metadata, self.source_fields))
        return f"CachedTensor({source_repr})"

    def __getattr__(self, name):
        if name in self.metadata:
            return self.metadata[name]
        else:
            raise AttributeError(
                f"{type(self).__name__} object has no attribute '{name}'"
            )

    def __tensor_flatten__(self):
        ctx = {
            "source_fields": self.source_fields,
            "extra_fields": self.extra_fields,
        }
        return [x for x in self.all_fields if self.metadata.get(x) is not None], ctx

    @staticmethod
    def __tensor_unflatten__(inner_tensors, meta, outer_size, outer_stride):
        return CachedTensor(inner_tensors, **meta)

    @classmethod
    def __torch_dispatch__(cls, op, types, args, kwargs):
        # Ensure that any registered ops are loaded
        import torch.nested._internal.ops

        # Doing any operation on a CachedTensor automatically unwraps and returns a non-CachedTensor
        # We can improve this to do smarter things, like automatically cache .diff(), .cumsum(), etc.
        if kwargs is None:
            kwargs = {}

        if op in _func_registry:
            return _func_registry[op](op, *args, **kwargs)

        unwrapped_args = pytree.tree_map_only(
            CachedTensor, lambda x: _get_source(x.metadata, x.source_fields), args
        )
        unwrapped_kwargs = pytree.tree_map_only(
            CachedTensor, lambda x: _get_source(x.metadata, x.source_fields), kwargs
        )
        return op(*unwrapped_args, **unwrapped_kwargs)


torch.serialization.add_safe_globals([CachedTensor])


_func_registry = {}


# Note: [ CacheTensor open registry ]
#
# Registering this decorator for an op allows CachedTensor's torch dispatch to
# redirect calls to that op to your own implementations.
# For NestedTensor we rely on this behavior for factory functions.
def register_cached_tensor_func(aten_op):
    def wrapper(func):
        _func_registry[aten_op] = func
        return func

    return wrapper


# NestedTensor-specific helpers
def _make_cached_tensor(metadata, target_field=None):
    from torch.nested._internal.nested_tensor import extra_fields, source_fields

    return CachedTensor(
        metadata,
        source_fields=source_fields,
        extra_fields=extra_fields,
        target_field=target_field,
    )

def make_cached_tensor_with_offsets(offsets):
    prefix = "_host" if offsets.is_cpu else "_device"
    metadata = {
        f"{prefix}_offsets": offsets
    }
    return _make_cached_tensor(metadata)
