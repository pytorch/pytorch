import logging
import os
import sys
from collections.abc import Callable
from functools import lru_cache


@lru_cache
def is_dev() -> bool:
    return os.environ.get("DEV") is not None


@lru_cache
def no_color() -> bool:
    return os.environ.get("NO_COLOR", "") != ""


def _ansi(code: int) -> str:
    return f"\033[{code}m"


class text:
    red = _ansi(31)
    green = _ansi(32)
    yellow = _ansi(33)
    cyan = _ansi(36)
    white = _ansi(37)


class back:
    red = _ansi(41)


class style:
    dim = _ansi(2)
    normal = _ansi(22)


def make(
    message: str,
    text: str | None = None,
    back: str | None = None,
    style: str | None = None,
) -> str:
    if no_color():
        return message
    codes = "".join(code for code in (text, back, style) if code is not None)
    return f"{codes}{message}\033[0m"


_LEVEL_TO_COLOR: dict[int, Callable[[str], str]] = {
    logging.CRITICAL: lambda m: make(m, text=text.white, back=back.red),
    logging.ERROR: lambda m: make(m, text=text.red),
    logging.WARNING: lambda m: make(m, text=text.yellow),
    logging.INFO: lambda m: make(m, text=text.green),
    logging.DEBUG: lambda m: make(m, text=text.cyan),
    logging.NOTSET: lambda m: make(m, style=style.normal),
}

_LEVEL_TO_NAME = {
    logging.CRITICAL: "c",
    logging.ERROR: "e",
    logging.WARNING: "w",
    logging.INFO: "i",
    logging.DEBUG: "d",
    logging.NOTSET: "_",
}
_LEVEL_NAME_WIDTH = max(len(name) for name in _LEVEL_TO_NAME.values())


class _Formatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        level_name = _LEVEL_TO_NAME.get(record.levelno, record.levelname)
        colorize = _LEVEL_TO_COLOR.get(record.levelno, lambda m: m)
        components = [
            make("%(asctime)s", style=style.dim),
            make("%(process)d", style=style.dim) if is_dev() else None,
            colorize(level_name.rjust(_LEVEL_NAME_WIDTH)),
            make("%(filename)s:%(lineno)d", style=style.dim) if is_dev() else None,
            "%(message)s",
        ]
        formatter = logging.Formatter(
            fmt=" ".join(c for c in components if c is not None),
            datefmt="%H:%M:%S",
        )
        return formatter.format(record)


_handler = logging.StreamHandler(stream=sys.stderr)
_handler.setFormatter(_Formatter())

log = logging.getLogger("tools.sandbox")
log.setLevel(logging.DEBUG if is_dev() else logging.INFO)
log.addHandler(_handler)
log.propagate = False
