"""ramjet's log, on stderr so your command's output on stdout stays untouched. Each
line starts with "ramjet", the time and the level, except the builder's output: logged
with extra=VERBATIM, it passes through as the builder printed it."""

import logging
import sys


# SGR codes: the numbers in ANSI escapes that set color and style
_RESET = 0
_DIM = 2
_RED = 31
_GREEN = 32
_YELLOW = 33
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
VERBATIM = {"verbatim": True}

# Not the root logger, which gets the debug messages of Modal's gRPC libraries
log = logging.getLogger("ramjet")
log.addHandler(logging.NullHandler())  # quiet until setup_logger, as in tests


def _ansi(message: str, *codes: int) -> str:
    return f"\033[{';'.join(map(str, codes))}m{message}\033[{_RESET}m"


def red(text: str) -> str:
    return _ansi(text, _RED)


class _Formatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        if getattr(record, "verbatim", False):
            return record.getMessage()
        name, color = _LEVELS[record.levelno]
        fields = [
            _ansi("ramjet", _DIM),
            _ansi("%(asctime)s", _DIM),
            _ansi(name.rjust(_LEVEL_WIDTH), *color),
            "%(message)s",
        ]
        return logging.Formatter(" ".join(fields), datefmt="%H:%M:%S").format(record)


def setup_logger() -> None:
    """Log at info until the caller sets another level."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_Formatter())
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    log.propagate = False
