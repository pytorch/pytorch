"""
Python polyfills for _io
"""

from __future__ import annotations

import _io

import sys
import warnings

from ..decorators import substitute_in_graph


__all__ = ["text_encoding"]


# Copied from Lib/_pyio.py in the standard library
# pyrefly: ignore [bad-argument-type]
@substitute_in_graph(_io.text_encoding, can_constant_fold_through=True)
def text_encoding(encoding: str | None, stacklevel: int = 2, /) -> str:
    if encoding is None:
        encoding = "utf-8" if sys.flags.utf8_mode else "locale"
        if sys.flags.warn_default_encoding:
            warnings.warn(
                "'encoding' argument not specified.",
                EncodingWarning,
                stacklevel + 1,
            )
    return encoding
