import typing as t


class CoverageRecord(t.NamedTuple):
    filepath: str
    covered_lines: t.List[int]
    uncovered_lines: t.Optional[t.List[int]] = None

    def to_dict(self) -> t.Dict[str, t.Any]:
        return {
            "filepath": self.filepath,
            "covered_lines": self.covered_lines,
            "uncovered_lines": self.uncovered_lines,
        }
