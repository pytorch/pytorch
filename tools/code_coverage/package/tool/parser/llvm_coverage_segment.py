from __future__ import annotations

from typing import NamedTuple


class LlvmCoverageSegment(NamedTuple):
    line: int
    col: int
    segment_count: int
    has_count: int
    is_region_entry: int
    is_gap_entry: int | None

    @property
    def has_coverage(self) -> bool:
        return self.segment_count > 0

    @property
    def is_executable(self) -> bool:
        return self.has_count > 0

    def get_coverage(
        self, prev_segment: LlvmCoverageSegment
    ) -> tuple[list[int], list[int]]:
        # Code adapted from testpilot.testinfra.runners.gtestcoveragerunner.py
        if not prev_segment.is_executable:
            return [], []

        # this segment ends at the line if col == 1
        # (so segment effectively ends on the line) and
        # line+1 if col is > 1 (so it touches at least some part of last line).
        end_of_segment = self.line if self.col == 1 else self.line + 1
        lines_range = list(range(prev_segment.line, end_of_segment))
        return (lines_range, []) if prev_segment.has_coverage else ([], lines_range)


def parse_segments(raw_segments: list[list[int]]) -> list[LlvmCoverageSegment]:
    """
    Creates LlvmCoverageSegment from a list of lists in llvm export json.
    each segment is represented by 5-element array.
    """
    ret: list[LlvmCoverageSegment] = []
    for raw_segment in raw_segments:
        assert (
            len(raw_segment) == 5 or len(raw_segment) == 6
        ), "list is not compatible with llvmcom export:"
        " Expected to have 5 or 6 elements"
        if len(raw_segment) == 5:
            ret.append(
                LlvmCoverageSegment(
                    raw_segment[0],
                    raw_segment[1],
                    raw_segment[2],
                    raw_segment[3],
                    raw_segment[4],
                    None,
                )
            )
        else:
            ret.append(LlvmCoverageSegment(*raw_segment))

    return ret
