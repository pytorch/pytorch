import functools

import torch

from torch._inductor.cuda_properties import get_device_capability


@functools.lru_cache(None)
def has_triton() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        import triton

        return triton is not None and get_device_capability() >= (7, 0)
    except ImportError:
        return False
