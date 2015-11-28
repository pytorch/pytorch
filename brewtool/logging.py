"""Simple logging utilities.
"""
import sys


class Colors(object):
    """A simple class that wraps some color codes. Old good ASCII art stuff."""
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'  # Yellow
    FAIL = '\033[91m'  # Red
    ENDCOLOR = '\033[0m'


def _PrintColor(color, message, *args, **kwargs):
    if args == () and kwargs == {}:
        out = message
    else:
        out = message.format(*args, **kwargs)
    if color is None:
        print(out)
    else:
        print(color + out + Colors.ENDCOLOR)


def BuildPrint(message, *args, **kwargs):
    return _PrintColor(None, message, *args, **kwargs)


def BuildDebug(message, *args, **kwargs):
    return _PrintColor(Colors.OKBLUE, "DEBUG: " + message, *args, **kwargs)


def BuildLog(message, *args, **kwargs):
    return _PrintColor(Colors.OKGREEN, "LOG: " + message, *args, **kwargs)


def BuildWarning(message, *args, **kwargs):
    return _PrintColor(Colors.WARNING, "WARNING: " + message, *args, **kwargs)


def BuildFatal(message, *args, **kwargs):
    _PrintColor(Colors.FAIL, "FATAL: " + message, *args, **kwargs)
    _PrintColor(Colors.FAIL, "Exiting.")
    sys.exit(1)


def BuildFatalIf(predicate, message, *args, **kwargs):
    if predicate:
        BuildFatal(message, *args, **kwargs)
