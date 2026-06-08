from __future__ import annotations

import torch


LOW_PRECISION_FP_DTYPES = (torch.bfloat16, torch.float16)


def low_precision_autocast_enabled() -> bool:
    for device_type in torch._C._autocast_supported_devices():
        if (
            torch.is_autocast_enabled(device_type)
            and torch.get_autocast_dtype(device_type) in LOW_PRECISION_FP_DTYPES
        ):
            return True
    return False
