import inspect
import weakref
from typing import Any

"""
Helper utilities for Dynamo that do not require a full import of torch._dynamo.
The motivation for this file is to avoid circular dependencies.
"""

DYNAMO_FORCE_INLINE: Any = weakref.WeakKeyDictionary()


def force_inline(f):
    """Forces inline on all functions that have the same __code__ as f,
    bypassing things like the SKIPFILES check.

    This matters for lambdas and nested functions: each new instance
    has a different __code__.

    This API is a short-term hack to avoid design flaws in SKIPFILES:
    there was not an easy way to mark a single API in a file that is
    in SKIPFILES for inlining. We should rework the SKIPFILES design
    (and this API).
    """
    if not (inspect.isfunction(f) and hasattr(f, "__code__")):
        raise ValueError(
            "force_inline(f): Expected f to be a Python function with .__code__"
        )
    DYNAMO_FORCE_INLINE[f.__code__] = True
    return f
