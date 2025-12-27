# mypy: allow-untyped-defs
from __future__ import annotations

import sys
from typing import Any, TYPE_CHECKING


class _LazySympyModule:
    _sympy = None

    @classmethod
    def _load(cls):
        if cls._sympy is None:
            import sympy

            cls._sympy = sympy
        return cls._sympy

    def __getattr__(self, name: str) -> Any:
        return getattr(self._load(), name)

    def __repr__(self) -> str:
        if self._sympy is None:
            return "<lazy module 'sympy' (not yet loaded)>"
        return repr(self._sympy)


_lazy_sympy = _LazySympyModule()


def get_sympy():
    return _LazySympyModule._load()


def is_sympy_loaded() -> bool:
    return "sympy" in sys.modules


if TYPE_CHECKING:
    import sympy

    sympy = sympy  # noqa: F811
else:
    sympy = _lazy_sympy
