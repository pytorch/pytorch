from __future__ import annotations

import torch
from torch.utils._ordered_set import OrderedSet


LOW_PRECISION_FP_DTYPES = (torch.bfloat16, torch.float16)

# These pointwise ops are exact on low-precision inputs, so forcing an
# eager-style round trip only adds layout pressure without changing values.
LOW_PRECISION_POINTWISE_BARRIER_EXEMPT_OPS = OrderedSet(
    [
        torch.ops.aten.relu.default,
    ]
)


def low_precision_autocast_enabled() -> bool:
    for device_type in torch._C._autocast_supported_devices():
        if (
            torch.is_autocast_enabled(device_type)
            and torch.get_autocast_dtype(device_type) in LOW_PRECISION_FP_DTYPES
        ):
            return True
    return False


def needs_low_precision_pointwise_barrier(func: object) -> bool:
    return func not in LOW_PRECISION_POINTWISE_BARRIER_EXEMPT_OPS
