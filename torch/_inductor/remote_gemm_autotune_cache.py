import asyncio
from typing import Any, TypeVar

import torch._inductor.config as config
from torch._inductor import ir


_T = TypeVar("_T")


def gen_best_config(
    mm_type: str, mats: list[ir.StorageBox], **kwargs: Any
) -> asyncio.Task[_T]:
    """
    Generate the best GEMM autotune config for the given matrices.
    """
    if config.is_fbcode():
        from torch._inductor.fb.remote_gemm_autotune_cache import gen_best_config

        return gen_best_config(mm_type, mats, **kwargs)
    else:
        raise NotImplementedError("Function gen_best_config is not yet implemented")
