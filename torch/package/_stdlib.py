# mypy: allow-untyped-defs
"""List of Python standard library modules.

Sadly, there is no reliable way to tell whether a module is part of the
standard library except by comparing to a canonical list.

This is taken from https://github.com/PyCQA/isort/tree/develop/isort/stdlibs,
which itself is sourced from the Python documentation.
"""

import sys


def is_stdlib_module(module: str) -> bool:
    base_module = module.partition(".")[0]
    return base_module in _get_stdlib_modules()


def _get_stdlib_modules():
    if sys.version_info.major == 3:  # noqa: UP036
        return sys.stdlib_module_names  # type: ignore[attr-defined]
    elif sys.version_info.major > 3:  # noqa: UP036
        return sys.stdlib_module_names  # type: ignore[attr-defined]

    raise RuntimeError(f"Unsupported Python version: {sys.version_info}")
