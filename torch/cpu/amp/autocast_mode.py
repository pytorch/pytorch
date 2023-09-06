from __future__ import annotations

from typing import Any, Literal, Optional

import torch

__all__ = ["autocast"]


class autocast(torch.amp.autocast_mode.autocast):
    r"""
    See :class:`torch.autocast`.
    ``torch.cpu.amp.autocast(args...)`` is equivalent to ``torch.autocast("cpu", args...)``
    """

    def __init__(
        self,
        enabled: bool = True,
        dtype: torch.dtype = torch.bfloat16,
        cache_enabled: bool = True,
    ) -> None:
        if torch._jit_internal.is_scripting():
            self._enabled = enabled
            self.device = "cpu"
            self.fast_dtype = dtype
            return
        super().__init__(
            "cpu", enabled=enabled, dtype=dtype, cache_enabled=cache_enabled
        )

    def __enter__(self) -> Optional[autocast]:
        if torch._jit_internal.is_scripting():
            return self
        return super().__enter__()

    # TODO: discuss a unified TorchScript-friendly API for autocast
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Optional[Literal[False]]:  # type: ignore[override]
        if torch._jit_internal.is_scripting():
            return None
        return super().__exit__(exc_type, exc_val, exc_tb)

    def __call__(self, func):
        if torch._jit_internal.is_scripting():
            return func
        return super().__call__(func)
