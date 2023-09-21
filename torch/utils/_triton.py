import functools

import torch


@functools.lru_cache(None)
def has_triton() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        import triton

        return triton is not None and torch.cuda.get_device_capability() >= (7, 0)
    except ImportError:
        return False
