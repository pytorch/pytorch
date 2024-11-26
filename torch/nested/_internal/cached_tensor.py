import torch
from torch.fx.operator_schemas import normalize_function
from torch.nested._internal.offload_tensor import register_tensor, try_get_int
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
    ):
        assert source_fields is not None
        assert any(
            metadata.get(k) is not None for k in source_fields
        ), f"CachedTensor: At least one of {source_fields} must be passed"

        # Tensor's metadata is the first non-None source field
        source = _get_source(metadata, source_fields)
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

    def __init__(self, metadata: dict, source_fields=None, extra_fields=()):
        # All source fields are registered, last non-None source field is the inner_id
        self.inner_id = None
        for k, v in metadata.items():
            if k in source_fields:
                if try_get_int(v) is None:
                    self.inner_id = register_tensor(v)
                else:
                    self.inner_id = try_get_int(v)
        # Why is inner_id none?
        assert self.inner_id is not None
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
        from torch.fx.experimental.proxy_tensor import maybe_enable_thunkify

        # Doing any operation on a CachedTensor automatically unwraps and returns a non-CachedTensor
        # We can improve this to do smarter things, like automatically cache .diff(), .cumsum(), etc.
        if kwargs is None:
            kwargs = {}

        if op in _func_registry:
            with maybe_enable_thunkify():
                return _func_registry[op](op, *args, **kwargs)

        unwrapped_args = pytree.tree_map_only(
            CachedTensor, lambda x: _get_source(x.metadata, x.source_fields), args
        )
        unwrapped_kwargs = pytree.tree_map_only(
            CachedTensor, lambda x: _get_source(x.metadata, x.source_fields), kwargs
        )
        return op(*unwrapped_args, **unwrapped_kwargs)


_func_registry = {}


def register_func(aten_op):
    def wrapper(func):
        _func_registry[aten_op] = func
        return func

    return wrapper


@register_func(torch.ops.aten._nested_from_padded_tensor.default)
def _nested_from_padded_tensor_default(func, *args, **kwargs):
    # padded: t, metadata: t, ragged_idx: any?, sum_S: any?
    from torch.nested._internal.nested_tensor import NestedTensor

    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    if new_kwargs["ragged_idx"] != 1:
        raise RuntimeError(
            "_nested_from_padded_tensor(): only ragged_idx=1 supported for jagged layout"
        )

    padded, offsets = new_kwargs["padded"], new_kwargs["offsets"]

    # non-3D padded is not supported by the underlying FBGEMM kernel so do shape gymnastics
    padded_shape = padded.shape
    if padded.dim() > 3:
        padded = padded.flatten(start_dim=2)
    elif padded.dim() < 3:
        padded = padded.unsqueeze(-1)

    # NB: The CUDA kernel for padded dense -> jagged conversion does not support
    # integer / bool types; work around this by casting to half.
    is_bool = padded.dtype is torch.bool
    if is_bool and padded.is_cuda:
        padded = padded.to(torch.half)
    values = torch.ops.aten._padded_dense_to_jagged_forward(
        padded, [offsets], new_kwargs["sum_S"]
    )
    if is_bool and values.is_cuda:
        values = values.to(torch.bool)

    # shape gymnastics part 2
    if len(padded_shape) > 3:
        values = values.unflatten(-1, padded_shape[2:])
    elif len(padded_shape) < 3:
        values = values.squeeze(-1)

    return NestedTensor(
        values,
        new_kwargs["metadata"],
        _ragged_idx=new_kwargs["ragged_idx"],
    )


@register_func(torch.ops.aten._nested_view_from_jagged.default)
def _nested_view_from_jagged_default(func, *args, **kwargs):
    # values: t, metadata: t, ragged_idx: any?
    from torch.nested._internal.nested_tensor import NestedTensor

    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )
    return NestedTensor(
        new_kwargs["input"],
        new_kwargs["metadata"],
        _ragged_idx=new_kwargs["ragged_idx"],
    )


@torch._dynamo.allow_in_graph
def make_cached_tensor(metadata):
    from torch.nested._internal.nested_tensor import extra_fields, source_fields

    return CachedTensor(
        metadata,
        source_fields=source_fields,
        extra_fields=extra_fields,
    )
