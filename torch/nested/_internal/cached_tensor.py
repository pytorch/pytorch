from contextlib import contextmanager
from typing import *  # noqa: F403

import torch
from torch.utils import _pytree as pytree


class CachedTensor(torch.Tensor):
    metadata: Dict[str, Optional[torch.Tensor]]
    source_field: str

    # Tensor subclass wrapping a dict of tensors, whose shape, dtype, device, etc.
    # is determined by a "source" tensor in the dict (specified by the user
    # during construction).
    #
    # This class is not super useful on its own because by default, performing any
    # operations on it will be as if you first unwrapped the CachedTensor and then
    # performed the operation on the plain source tensor (a plain tensor is also returned).
    # To leverage the extra metadata, you must register an op to perform the special logic
    # you want via register_cached_tensor_func.
    #
    # When used this way (1) it is a convenient way to keep around metadata
    # related to a tensor, without having to laboriously thread those extra metadata
    # around, e.g. through aten signatures. (2) allows one to trigger custom
    # __torch_dispatch__ logic to construct tensor subclasses (which can be tricky to do
    # otherwise because the subclass constructor op itself usually does not take the
    # subclass itself as input!). See NestedTensor for an example.

    @staticmethod
    @torch._disable_dynamo  # type: ignore[misc]
    def __new__(
        cls,
        metadata: Dict[str, Optional[torch.Tensor]],
        source_field: str,
    ) -> "CachedTensor":
        source = metadata.get(source_field)
        assert source is not None
        shape = source.shape
        kwargs = {}
        kwargs["strides"] = source.stride()
        kwargs["storage_offset"] = source.storage_offset()  # type: ignore[assignment]
        kwargs["device"] = source.device  # type: ignore[assignment]
        kwargs["layout"] = source.layout  # type: ignore[assignment]
        kwargs["requires_grad"] = source.requires_grad  # type: ignore[assignment]
        kwargs["dtype"] = source.dtype  # type: ignore[assignment]
        out = torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

        out.metadata = metadata
        out.source_field = source_field
        return out

    def __repr__(self) -> str:  # type: ignore[override]
        return f"CachedTensor({repr(self.metadata[self.source_field])})"

    def __getattr__(self, name: str) -> Optional[torch.Tensor]:
        if name not in self.metadata:
            raise AttributeError(
                f"{type(self).__name__} object has no attribute '{name}'"
            )
        return self.metadata[name]

    def __tensor_flatten__(self) -> Tuple[List[str], Dict[str, Any]]:
        ctx = {
            "source_field": self.source_field,
            "all_fields": list(self.metadata.keys()),
        }
        return [k for k, v in self.metadata.items() if v is not None], ctx

    @staticmethod
    def __tensor_unflatten__(
        inner_tensors: Dict, meta: Dict, outer_size: Any, outer_stride: Any
    ) -> "CachedTensor":
        inner_tensors = {k: inner_tensors.get(k) for k in meta["all_fields"]}
        return CachedTensor(inner_tensors, source_field=meta["source_field"])

    @classmethod
    def __torch_dispatch__(
        cls, op: Any, types: Any, args: Any, kwargs: Any
    ) -> torch.Tensor:
        # Ensure that any registered ops are loaded
        import torch.nested._internal.ops  # noqa: F401

        if kwargs is None:
            kwargs = {}

        if op in _func_registry:
            return _func_registry[op](op, *args, **kwargs)

        # By default, doing any operation on a CachedTensor automatically unwraps and
        # returns a non-CachedTensor
        unwrapped_args = pytree.tree_map_only(
            CachedTensor,
            lambda x: x.metadata[x.source_field],
            args,
        )
        unwrapped_kwargs = pytree.tree_map_only(
            CachedTensor,
            lambda x: x.metadata[x.source_field],
            kwargs,
        )
        return op(*unwrapped_args, **unwrapped_kwargs)


torch.serialization.add_safe_globals([CachedTensor])


_func_registry: Dict[Any, Callable] = {}


@contextmanager
def set_func_registry(registry: Dict[Any, Callable]) -> Generator:
    global _func_registry
    old_registry = _func_registry
    _func_registry = registry
    try:
        yield
    finally:
        _func_registry = old_registry


# Note: [ CacheTensor open registry ]
#
# Registering this decorator for an op allows CachedTensor's torch dispatch to
# redirect calls to that op to your own implementations.
# For NestedTensor we rely on this behavior for factory functions.
def register_cached_tensor_func(aten_op: Any) -> Callable:
    def wrapper(func: Callable) -> Callable:
        _func_registry[aten_op] = func
        return func

    return wrapper
