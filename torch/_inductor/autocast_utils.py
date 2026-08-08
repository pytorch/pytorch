from __future__ import annotations

import contextlib
import contextvars
from typing import TYPE_CHECKING

import torch
from torch.utils._ordered_set import OrderedSet


if TYPE_CHECKING:
    from collections.abc import Generator


LOW_PRECISION_FP_DTYPES = (torch.bfloat16, torch.float16)
LOW_PRECISION_CAST_BOUNDARY = "inductor_low_precision_cast_boundary"
LOW_PRECISION_CAST_OPS = OrderedSet(
    [
        torch.ops.aten._to_copy.default,
        torch.ops.aten.to.device,
        torch.ops.aten.to.dtype,
        torch.ops.aten.to.dtype_layout,
        torch.ops.aten.to.other,
        torch.ops.aten.type_as.default,
        torch.ops.prims.convert_element_type.default,
    ]
)
DTYPE_CAST_METHOD_DTYPES = {
    "bfloat16": torch.bfloat16,
    "bool": torch.bool,
    "byte": torch.uint8,
    "cdouble": torch.complex128,
    "cfloat": torch.complex64,
    "chalf": torch.complex32,
    "char": torch.int8,
    "double": torch.float64,
    "float": torch.float32,
    "half": torch.float16,
    "int": torch.int32,
    "long": torch.int64,
    "short": torch.int16,
}
LOW_PRECISION_CAST_METHODS = OrderedSet(
    [
        *DTYPE_CAST_METHOD_DTYPES,
        "to",
        "type",
        "type_as",
    ]
)
_FORCE_LOW_PRECISION_POINTWISE_BARRIERS: contextvars.ContextVar[bool] = (
    contextvars.ContextVar("force_low_precision_pointwise_barriers", default=False)
)

# These pointwise ops are exact on low-precision inputs, so forcing an
# eager-style round trip only adds layout pressure without changing values.
LOW_PRECISION_POINTWISE_BARRIER_EXEMPT_OPS = OrderedSet(
    [
        torch.ops.aten.relu.default,
    ]
)


def low_precision_autocast_enabled() -> bool:
    # _is_any_autocast_enabled does not currently include MAIA or MPS.
    if not torch._C._is_any_autocast_enabled() and not any(
        torch.is_autocast_enabled(device_type) for device_type in ("maia", "mps")
    ):
        return False

    for device_type in torch._C._autocast_supported_devices():
        if (
            torch.is_autocast_enabled(device_type)
            and torch.get_autocast_dtype(device_type) in LOW_PRECISION_FP_DTYPES
        ):
            return True
    return False


def low_precision_pointwise_barriers_enabled() -> bool:
    return (
        _FORCE_LOW_PRECISION_POINTWISE_BARRIERS.get()
        or low_precision_autocast_enabled()
    )


@contextlib.contextmanager
def force_low_precision_pointwise_barriers() -> Generator[None, None, None]:
    token = _FORCE_LOW_PRECISION_POINTWISE_BARRIERS.set(True)
    try:
        yield
    finally:
        _FORCE_LOW_PRECISION_POINTWISE_BARRIERS.reset(token)


def needs_low_precision_pointwise_barrier(func: object) -> bool:
    return (
        isinstance(func, torch._ops.OpOverload)
        # Eager cast boundaries other than the prim are not tagged pointwise.
        and (torch.Tag.pointwise in func.tags or func in LOW_PRECISION_CAST_OPS)
        and func not in LOW_PRECISION_POINTWISE_BARRIER_EXEMPT_OPS
    )
