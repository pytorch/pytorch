# Tests that top-level ClassVar is not allowed

from __future__ import annotations

from typing import ClassVar

wrong: ClassVar[int] = 1
