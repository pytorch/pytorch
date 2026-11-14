"""ramjet's log, on stderr so your command's output on stdout stays untouched. Each
line starts with "ramjet", the time and the level, except the builder's output: logged
with extra=VERBATIM, it passes through as the builder printed it."""

import contextlib
import logging
import os
import sys
from collections.abc import Iterator
from typing import Any, TextIO


# SGR codes: the numbers in ANSI escapes that set color and style
_RESET = 0
_BOLD = 1
_DIM = 2
_RED = 31
_GREEN = 32
_YELLOW = 33
_MAGENTA = 35
_CYAN = 36
# Each level's name and color, least severe first
_LEVELS = {
    logging.DEBUG: ("debug", (_CYAN,)),
    logging.INFO: ("info", (_GREEN,)),
    logging.WARNING: ("warn", (_YELLOW,)),
    logging.ERROR: ("error", (_RED,)),
}
_LEVEL_WIDTH = max(len(name) for name, _ in _LEVELS.values())
# The levels by name: ramjet's --log-level choices
LEVEL_NAMES = {name: level for level, (name, _) in _LEVELS.items()}
# The columns a line's start takes, before its message
MARGIN = len("ramjet 00:00:00 ") + _LEVEL_WIDTH + 1
VERBATIM = {"verbatim": True}
# With these as extra: a line that goes on under the last one's message, without a
# start of its own, and a line that rings the terminal's bell. A continued line may
# also give a "margin": a name for it, which ends where the level does
CONTINUED = {"continued": True}
RING = {"ring": True}

# Not the root logger, which gets the debug messages of Modal's gRPC libraries
log = logging.getLogger("ramjet")
log.addHandler(logging.NullHandler())  # quiet until setup_logger, as in tests


def _ansi(message: str, *codes: int) -> str:
    return f"\033[{';'.join(map(str, codes))}m{message}\033[{_RESET}m"


def red(text: str) -> str:
    return _ansi(text, _RED)


def green(text: str) -> str:
    return _ansi(text, _GREEN)


def magenta(text: str) -> str:
    return _ansi(text, _MAGENTA)


def bold(text: str) -> str:
    return _ansi(text, _BOLD)


def dim(text: str) -> str:
    return _ansi(text, _DIM)


class _Formatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        if getattr(record, "verbatim", False):
            return record.getMessage()
        if getattr(record, "continued", False):
            name = getattr(record, "margin", "")
            pad = " " * (MARGIN - 1 - len(name))
            return f"{pad}{dim(name) if name else ''} {record.getMessage()}"
        name, color = _LEVELS[record.levelno]
        fields = [
            dim("ramjet"),
            dim("%(asctime)s"),
            _ansi(name.rjust(_LEVEL_WIDTH), *color),
            "%(message)s",
        ]
        line = logging.Formatter(" ".join(fields), datefmt="%H:%M:%S").format(record)
        ring = getattr(record, "ring", False) and sys.stderr.isatty()
        return f"\a{line}" if ring else line


# Any: for a while, test's progress stands in for stderr (through, below)
_HANDLER: "logging.StreamHandler[Any]" = logging.StreamHandler(sys.stderr)
_HANDLER.setFormatter(_Formatter())


def setup_logger() -> None:
    """Log at info until the caller sets another level."""
    log.addHandler(_HANDLER)
    log.setLevel(logging.INFO)
    log.propagate = False


def terminal() -> TextIO | None:
    """The terminal that shows the log's lines at info, if one does."""
    shown = _HANDLER in log.handlers and log.isEnabledFor(logging.INFO)
    plain = os.environ.get("TERM") == "dumb"  # it takes no control sequences
    return sys.stderr if shown and sys.stderr.isatty() and not plain else None


@contextlib.contextmanager
def through(stream: Any) -> Iterator[None]:
    """For the block, the log's lines are written to the stream, not to stderr."""
    stderr = _HANDLER.setStream(stream)
    try:
        yield
    finally:
        _HANDLER.setStream(stderr)
