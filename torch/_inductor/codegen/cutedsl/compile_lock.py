"""Process-wide lock for in-process CuTeDSL compilation (see
torch._native.cutedsl.compile_lock, which defines it)."""

from torch._native.cutedsl.compile_lock import CUTEDSL_COMPILE_LOCK


__all__ = ["CUTEDSL_COMPILE_LOCK"]
