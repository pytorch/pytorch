"""Utils for Meta Triton autoWS."""

from __future__ import annotations

import functools


@functools.cache
def has_meta_ws() -> bool:
    """Whether Meta Triton autoWS is available."""
    try:
        from triton import knobs
    except ImportError:
        return False
    return hasattr(getattr(knobs, "nvidia", None), "use_meta_ws")


def meta_ws_enabled() -> bool:
    """Whether Meta Triton autoWS is available and enabled via its knob.

    Reads the live ``knobs.nvidia.use_meta_ws`` value (backed by
    TRITON_USE_META_WS) rather than the env var directly, so it honors the knob
    set programmatically or within a ``knobs.nvidia.scope()``.
    """
    if not has_meta_ws():
        return False
    from triton import knobs

    return bool(knobs.nvidia.use_meta_ws)
