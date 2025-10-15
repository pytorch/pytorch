# Used by test_typing to verify that Final wrapped in ForwardRef works.

from __future__ import annotations

from typing import Final

name: Final[str] = "final"

class MyClass:
    value: Final = 3000
