from typing import Any, Dict, Callable, TypeVar
import textwrap

_BACK_COMPAT_OBJECTS : Dict[Any, None] = {}
_MARKED_WITH_COMPATIBILITY : Dict[Any, None] = {}

_T = TypeVar("_T")

def compatibility(is_backward_compatible : bool) -> Callable[[_T], _T]:
    if is_backward_compatible:

        def mark_back_compat(fn: _T) -> _T:
            docstring = textwrap.dedent(getattr(fn, '__doc__', None) or '')
            docstring += """
.. note::
    Backwards-compatibility for this API is guaranteed.
"""
            fn.__doc__ = docstring
            _BACK_COMPAT_OBJECTS.setdefault(fn)
            _MARKED_WITH_COMPATIBILITY.setdefault(fn)
            return fn

        return mark_back_compat
    else:

        def mark_not_back_compat(fn: _T) -> _T:
            docstring = textwrap.dedent(getattr(fn, '__doc__', None) or '')
            docstring += """
.. warning::
    This API is experimental and is *NOT* backward-compatible.
"""
            fn.__doc__ = docstring
            _MARKED_WITH_COMPATIBILITY.setdefault(fn)
            return fn

        return mark_not_back_compat
