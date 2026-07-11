# Copyright (c) 2025, Tri Dao.

"""Unified FlashAttention logging controlled by a single ``FA_LOG_LEVEL`` env var.

Host-side messages go through Python ``logging`` (logger name ``flash_attn``).
A default ``StreamHandler`` is attached automatically when ``FA_LOG_LEVEL >= 1``
so that standalone scripts get output without extra setup; applications that
configure their own logging can remove or replace it via the standard API.

FA_LOG_LEVEL mapping::

    0  off       nothing logged
    1  host      host-side summaries only (no kernel printf)
    2  kernel    host + curated kernel traces
    3  max       host + all kernel traces (noisy, perf hit)

Set via environment variable::

    FA_LOG_LEVEL=1 python train.py

Device-side ``cute.printf`` calls are compile-time eliminated via
``cutlass.const_expr`` when the log level is below the callsite threshold,
so there is zero performance cost when device logging is off.
Changing the log level after kernel compilation requires a recompile
(the level participates in the forward compile key).
"""

import logging
import os
import sys

import cutlass.cute as cute
from cutlass import const_expr

_LOG_LEVEL_NAMES = {"off": 0, "host": 1, "kernel": 2, "max": 3}


def _parse_log_level(raw: str) -> int:
    if raw in _LOG_LEVEL_NAMES:
        return _LOG_LEVEL_NAMES[raw]
    try:
        level = int(raw)
    except ValueError:
        return 0
    return max(0, min(level, 3))


_fa_log_level: int = _parse_log_level(os.environ.get("FA_LOG_LEVEL", "0"))

_logger = logging.getLogger("flash_attn")
_logger.addHandler(logging.NullHandler())
_default_handler: logging.Handler | None = None


def _configure_default_handler() -> None:
    global _default_handler
    if _fa_log_level >= 1:
        if _default_handler is None:
            _default_handler = logging.StreamHandler(sys.stdout)
            _default_handler.setFormatter(logging.Formatter("[FA] %(message)s"))
            _logger.addHandler(_default_handler)
        _logger.setLevel(logging.DEBUG)
    else:
        if _default_handler is not None:
            _logger.removeHandler(_default_handler)
            _default_handler = None
        _logger.setLevel(logging.WARNING)


_configure_default_handler()


def get_fa_log_level() -> int:
    return _fa_log_level


def set_fa_log_level(level: int | str) -> None:
    """Set the FA log level programmatically.

    Host logging takes effect immediately.  Device logging changes only
    affect kernels compiled after this call (new compile-key selection).
    """
    global _fa_log_level
    if isinstance(level, str):
        level = _parse_log_level(level)
    _fa_log_level = max(0, min(int(level), 3))
    _configure_default_handler()


def fa_log(level: int, msg: str):
    if _fa_log_level >= level:
        _logger.info(msg)


def fa_printf(level: int, fmt, *args):
    if const_expr(_fa_log_level >= level):
        cute.printf(fmt, *args)
