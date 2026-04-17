"""Utility for issuing warnings that point to the caller outside of torch."""

import os
import sys
import warnings

# Compute the torch package directory once at import time.
# Used as the prefix for skip_file_prefixes on Python >= 3.12.
# Include both the raw and resolved (realpath) paths so that symlinked
# installations are handled correctly.
_here = os.path.dirname(__file__)
_TORCH_PREFIXES: tuple[str, ...] = tuple(
    {_here + os.sep, os.path.realpath(_here) + os.sep}
)


def warn(
    message: str | Warning,
    category: type[Warning] = UserWarning,
    stacklevel: int = 2,
) -> None:
    """Issue a warning that points to the caller outside of ``torch``.

    On Python >= 3.12, uses ``skip_file_prefixes`` to skip *all* frames
    inside the ``torch`` package regardless of nesting depth.  On older
    Pythons this falls back to the caller-supplied *stacklevel* (default 2,
    i.e. one frame above the direct call site).
    """
    if sys.version_info >= (3, 12):
        warnings.warn(
            message,
            category,
            stacklevel=stacklevel,
            skip_file_prefixes=_TORCH_PREFIXES,
        )
    else:
        warnings.warn(message, category, stacklevel=stacklevel)
