# mypy: allow-untyped-defs
import warnings
from contextlib import contextmanager

import torch
from torch.backends import __allow_nonbracketed_mutation


__all__ = ["is_available", "flags", "set_flags"]

_flag = True
_warned = False


def _warn_deprecated():
    global _warned
    if not _warned or torch.is_warn_always_enabled():
        warnings.warn(
            "torch.backends.nnpack is deprecated since PyTorch 2.16 and will be removed in 2.18. "
            "NNPACK is no longer part of PyTorch, so is_available() always returns False and "
            "set_flags()/flags() have no effect; remove these calls.",
            FutureWarning,
            stacklevel=3,
        )
        _warned = True


def is_available():
    r"""Return whether PyTorch is built with NNPACK support.

    .. deprecated:: 2.16
        NNPACK has been removed from PyTorch; this always returns ``False``.
    """
    _warn_deprecated()
    return False


def set_flags(_enabled):
    r"""Set if nnpack is enabled globally.

    .. deprecated:: 2.16
        NNPACK has been removed from PyTorch; this has no effect.
    """
    global _flag
    _warn_deprecated()
    orig_flags = (_flag,)
    _flag = _enabled
    return orig_flags


@contextmanager
def flags(enabled=False):
    r"""Context manager for setting if nnpack is enabled globally.

    .. deprecated:: 2.16
        NNPACK has been removed from PyTorch; this has no effect.
    """
    with __allow_nonbracketed_mutation():
        orig_flags = set_flags(enabled)
    try:
        yield
    finally:
        with __allow_nonbracketed_mutation():
            set_flags(orig_flags[0])
