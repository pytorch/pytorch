"""
Python polyfills for io
"""

from __future__ import annotations

import _io

import sys

from ..decorators import substitute_in_graph


__all__ = ["text_encoding"]


# Copied from _pyio.py in the standard library
# pyrefly: ignore [bad-argument-type]
@substitute_in_graph(_io.text_encoding, can_constant_fold_through=True)
def text_encoding(encoding: str | None, stacklevel: int = 2) -> str:
    if encoding is not None:
        return encoding
    encoding = "utf-8" if sys.flags.utf8_mode else "locale"
    if sys.flags.warn_default_encoding:
        import warnings

        warnings.warn(
            "'encoding' argument not specified.",
            EncodingWarning,
            stacklevel + 1,
        )
    return encoding
