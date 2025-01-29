from contextlib import contextmanager
from typing import *  # noqa: F403

import torch


class DictTensor(torch.Tensor):
    metadata: Dict[str, Optional[torch.Tensor]]

    # Tensor subclass wrapping a dict of tensors.
    #
    # This tensor does not support any operations on it by default.
    # To leverage the extra metadata, you must register an op to perform the special logic
    # you want via register_dict_tensor_func.
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
    ) -> "DictTensor":
        shape = (0,)
        kwargs = {}
        kwargs["strides"] = (1,)
        kwargs["storage_offset"] = 0  # type: ignore[assignment]
        kwargs["device"] = "cpu" # type: ignore[assignment]
        kwargs["layout"] = torch.strided  # type: ignore[assignment]
        kwargs["requires_grad"] = False  # type: ignore[assignment]
        kwargs["dtype"] = torch.int64  # type: ignore[assignment]
        out = torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

        out.metadata = metadata
        return out

    def __repr__(self) -> str:  # type: ignore[override]
        return f"DictTensor(metadata={self.metadata})"

    def __getattr__(self, name: str) -> Optional[torch.Tensor]:
        if name not in self.metadata:
            raise AttributeError(
                f"{type(self).__name__} object has no attribute '{name}'"
            )
        return self.metadata[name]

    def __tensor_flatten__(self) -> Tuple[List[str], Dict[str, Any]]:
        ctx = {
            "all_fields": list(self.metadata.keys()),
        }
        return [k for k, v in self.metadata.items() if v is not None], ctx

    @staticmethod
    def __tensor_unflatten__(
        inner_tensors: Dict, meta: Dict, outer_size: Any, outer_stride: Any
    ) -> "DictTensor":
        inner_tensors = {k: inner_tensors.get(k) for k in meta["all_fields"]}
        return DictTensor(inner_tensors)

    @classmethod
    def __torch_dispatch__(
        cls, op: Any, types: Any, args: Any, kwargs: Any
    ) -> torch.Tensor:
        # Ensure that any registered ops are loaded
        import torch.nested._internal.ops  # noqa: F401

        if kwargs is None:
            kwargs = {}

        if op is torch.ops.aten.detach.default:
            # detach is needed for torch.compile
            return args[0]

        if op in _func_registry:
            return _func_registry[op](op, *args, **kwargs)

        # By default, doing any operation on a DictTensor errors
        raise NotImplementedError(f"DictTensor does not support for {op}")


torch.serialization.add_safe_globals([DictTensor])


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
# Registering this decorator for an op allows DictTensor's torch dispatch to
# redirect calls to that op to your own implementations.
# For NestedTensor we rely on this behavior for factory functions.
def register_dict_tensor_func(aten_op: Any) -> Callable:
    def wrapper(func: Callable) -> Callable:
        if aten_op in _func_registry:
            raise RuntimeError(
                f"Attempted to register {func} for {aten_op}, but {aten_op} is already registered to {_func_registry[aten_op]}"
            )
        _func_registry[aten_op] = func
        return func

    return wrapper
