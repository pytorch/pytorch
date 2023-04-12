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
