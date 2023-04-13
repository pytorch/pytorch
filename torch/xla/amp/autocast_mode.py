import torch
from typing import Any

__all__ = ["autocast"]

class autocast(torch.amp.autocast_mode.autocast):
    r"""
    See :class:`torch.autocast`.
    ``torch.xla.amp.autocast(args...)`` is equivalent to ``torch.autocast("xla", args...)``
    """
    def __init__(self, enabled : bool = True, dtype : torch.dtype = torch.bfloat16, cache_enabled : bool = True):
        super().__init__("xla", enabled=enabled, dtype=dtype, cache_enabled=cache_enabled)

    def __enter__(self):
        return super().__enter__()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):  # type: ignore[override]
        return super().__exit__(exc_type, exc_val, exc_tb)

    def __call__(self, func):
        return super().__call__(func)
