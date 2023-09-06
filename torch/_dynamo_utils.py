import weakref
from typing import Any

"""
Helper utilities for Dynamo that do not require a full import of torch._dynamo.
The motivation for this file is to avoid circular dependencies.
"""

DYNAMO_FORCE_INLINE: Any = weakref.WeakKeyDictionary()


def compiler_force_inline(f):
    """Forces inline on all functions that have the same __code__ as f,
    bypassing things like the SKIPFILES check.

    This matters for lambdas and nested functions: each new instance
    has a different __code__.

    This API is a short-term hack to avoid design flaws in SKIPFILES:
    there was not an easy way to mark a single API in a file that is
    in SKIPFILES for inlining. We should rework the SKIPFILES design
    (and this API).
    """
    DYNAMO_FORCE_INLINE[f.__code__] = True
    return f


def compiler_should_force_inline(func):
    return func.get_code() in DYNAMO_FORCE_INLINE
