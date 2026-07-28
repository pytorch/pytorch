"""Utils for Meta Triton autoWS."""

from __future__ import annotations

import functools
import inspect


@functools.cache
def has_meta_ws() -> bool:
    """Whether Meta Triton autoWS is available."""
    try:
        from triton import knobs
    except ImportError:
        return False
    return hasattr(getattr(knobs, "nvidia", None), "use_meta_ws")


@functools.cache
def has_two_ctas() -> bool:
    """Whether tl.dot supports two_ctas. It has no knob, so probe the signature
    directly; the meta-WS knob alone does not imply tl.dot accepts it."""
    try:
        import triton.language as tl
    except ImportError:
        return False
    return "two_ctas" in inspect.signature(tl.dot).parameters
