from typing import Any, Dict, List, Set

from .coverage_record import CoverageRecord


class GcovCoverageParser:
    """
        Accepts a parsed json produced by gcov --json-format -- typically,
        representing a single C++ test and produces a list
        of CoverageRecord(s).
    """

    def __init__(self, llvm_coverage: Dict[str, Any]) -> None:
        self._llvm_coverage = llvm_coverage

    @staticmethod
    def _skip_coverage(path: str) -> bool:
        """
            Returns True if file path should not be processed.
            This is repo-specific and only makes sense for the current state of
            ovrsource.
        """
        if "third-party" in path:
            return True
        return False

    def parse(self) -> List[CoverageRecord]:
        # The JSON format is described in the gcov source code
        # https://gcc.gnu.org/onlinedocs/gcc/Invoking-Gcov.html
        records: List[CoverageRecord] = []
        for file_info in self._llvm_coverage["files"]:
            filepath = file_info["file"]
            if self._skip_coverage(filepath):
                continue
            # parse json file
            covered_lines: Set[int] = set()
            uncovered_lines: Set[int] = set()
            for line in file_info["lines"]:
                line_number = line["line_number"]
                count = line["count"]
                if count == 0:
                    uncovered_lines.update([line_number])
                else:
                    covered_lines.update([line_number])

            records.append(
                CoverageRecord(filepath, sorted(covered_lines), sorted(uncovered_lines))
            )

        return records
