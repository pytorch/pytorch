import inspect
from collections.abc import Callable, Iterable
from typing import Any

import torch
import torch.utils._pytree as pytree

from .arg_path import parse_arg_path
from .dispatchless_op_overload import (
    DispatchlessOpOverload,
    DispatchlessOpOverloadPacket,
)
from .schema import encode_overload_name
from .tracing import trace_custom_op


# Cache the lookup, the lookup is ~50ns
_PROXY = torch._C._TorchDispatchModeKey.PROXY


class CustomOp:
    def __init__(
        self,
        qualname: str,
        fn: Callable[..., Any],
        mutates_args: Iterable[str] | str = (),
    ):
        self._ns, self._name = qualname.split("::")
        if "." in self._name:
            raise ValueError("custom_op qualnames cannot include overload names")
        self._fn = fn
        self._fake_fn: Callable[..., Any] | None = None
        self._sig = inspect.signature(fn)
        if mutates_args == "unknown":
            self._mutates = "unknown"
        else:
            self._mutates = tuple(parse_arg_path(path) for path in mutates_args)
            bad = {p.root for p in self._mutates} - self._sig.parameters.keys()
            if bad:
                raise ValueError(f"mutates_args names not in signature: {sorted(bad)}")
        self._cache: dict[tuple, DispatchlessOpOverload] = {}
        self._packet = DispatchlessOpOverloadPacket(self)
        setattr(getattr(torch.ops, self._ns), self._name, self._packet)

    def register_fake(self, fake_fn: Callable[..., Any]) -> Callable[..., Any]:
        """Register the kernel used to infer outputs under tracing_mode="fake"."""
        self._fake_fn = fake_fn
        return fake_fn

    def _get_or_create_overload(
        self,
        # The "(args) -> returns" portion of the schema, e.g.
        # "(Tensor(a0) t) -> (Tensor(a0), Tensor)".
        schema_body: str,
        in_spec: pytree.TreeSpec,
        # `None` out_spec means the op returns None (see tracing.py); we should
        # probably support None as a pytree type but that needs more maneuvering.
        out_spec: pytree.TreeSpec | None,
    ) -> Any:
        key = (schema_body, in_spec, out_spec)
        op = self._cache.get(key)
        if op is not None:
            return op

        overload_name = encode_overload_name(
            {
                "v": 1,
                "schema": schema_body,
                "in_spec": pytree.treespec_dumps(in_spec),
                "out_spec": None
                if out_spec is None
                else pytree.treespec_dumps(out_spec),
            }
        )
        # Avoid the dispatcher's 64-arg limit via our own kind of OpOverload.
        schema = torch._C.parse_schema(
            f"{self._ns}::{self._name}.{overload_name}{schema_body}"
        )
        op = DispatchlessOpOverload(self._packet, overload_name, schema, in_spec)
        self._cache[key] = op
        return op

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # This is the eager-hot path. Keep it as light as possible.
        mode = torch._C._get_dispatch_mode(_PROXY)
        if mode is not None:
            return trace_custom_op(self, mode, args, kwargs)
        return self._fn(*args, **kwargs)


def custom_op(
    qualname: str,
    *,
    mutates_args: Iterable[str] | str = (),
) -> Callable[[Callable[..., Any]], CustomOp]:
    """Decorate a Python function as a custom operator.

    Each input of the custom operator may be:
    - one of the supported leaf types (Tensor, SymInt/int, bool, float,
      str, dtype, device, ProcessGroup)
    - nested containers (list/tuple/dict/dataclasses) of the previous item

    Each output of the custom operator may be:
    - a symbolic leaf type (Tensor or SymInt)
    - nested containers of the previous item

    Please specify which Tensors are being mutated via `mutates_args`
    (these include paths to the tensors, e.g. "state['buf']" or "buffers[0]").
    """

    return lambda fn: CustomOp(qualname, fn, mutates_args)
