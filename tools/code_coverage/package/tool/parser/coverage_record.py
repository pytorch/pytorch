from __future__ import annotations

from typing import Any, NamedTuple


class CoverageRecord(NamedTuple):
    filepath: str
    covered_lines: list[int]
    uncovered_lines: list[int] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "filepath": self.filepath,
            "covered_lines": self.covered_lines,
            "uncovered_lines": self.uncovered_lines,
        }
