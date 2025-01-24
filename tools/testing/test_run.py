from __future__ import annotations

from copy import copy
from functools import total_ordering
from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Iterable


class TestRun:
    """
    TestRun defines the set of tests that should be run together in a single pytest invocation.
    It'll either be a whole test file or a subset of a test file.

    This class assumes that we won't always know the full set of TestClasses in a test file.
    So it's designed to include or exclude explicitly requested TestClasses, while having accepting
    that there will be an ambiguous set of "unknown" test classes that are not expliclty called out.
    Those manifest as tests that haven't been explicitly excluded.
    """

    test_file: str
    _excluded: frozenset[str]  # Tests that should be excluded from this test run
    _included: frozenset[
        str
    ]  # If non-empy, only these tests should be run in this test run

    # NB: Also the class is called TestRun, it's not a test class, so having this field set
    # will allow pytest to ignore this accordingly
    __test__ = False

    def __init__(
        self,
        name: str,
        excluded: Iterable[str] | None = None,
        included: Iterable[str] | None = None,
    ) -> None:
        if excluded and included:
            raise ValueError("Can't specify both included and excluded")

        ins = set(included or [])
        exs = set(excluded or [])

        if "::" in name:
            assert (
                not included and not excluded
            ), "Can't specify included or excluded tests when specifying a test class in the file name"
            self.test_file, test_class = name.split("::")
            ins.add(test_class)
        else:
            self.test_file = name

        self._excluded = frozenset(exs)
        self._included = frozenset(ins)

    @staticmethod
    def empty() -> TestRun:
        return TestRun("")

    def is_empty(self) -> bool:
        # Lack of a test_file means that this is an empty run,
        # which means there is nothing to run. It's the zero.
        return not self.test_file

    def is_full_file(self) -> bool:
        return not self._included and not self._excluded

    def included(self) -> frozenset[str]:
        return self._included

    def excluded(self) -> frozenset[str]:
        return self._excluded

    def get_pytest_filter(self) -> str:
        if self._included:
            return " or ".join(sorted(self._included))
        elif self._excluded:
            return f"not ({' or '.join(sorted(self._excluded))})"
        else:
            return ""

    def contains(self, test: TestRun) -> bool:
        if self.test_file != test.test_file:
            return False

        if self.is_full_file():
            return True  # self contains all tests

        if test.is_full_file():
            return False  # test contains all tests, but self doesn't

        # Does self exclude a subset of what test excludes?
        if test._excluded:
            return test._excluded.issubset(self._excluded)

        # Does self include everything test includes?
        if self._included:
            return test._included.issubset(self._included)

        # Getting to here means that test includes and self excludes
        # Does self exclude anything test includes? If not, we're good
        return not self._excluded.intersection(test._included)

    def __copy__(self) -> TestRun:
        return TestRun(self.test_file, excluded=self._excluded, included=self._included)

    def __bool__(self) -> bool:
        return not self.is_empty()

    def __repr__(self) -> str:
        r: str = f"RunTest({self.test_file}"
        r += f", included: {self._included}" if self._included else ""
        r += f", excluded: {self._excluded}" if self._excluded else ""
        r += ")"
        return r

    def __str__(self) -> str:
        if self.is_empty():
            return "Empty"

        pytest_filter = self.get_pytest_filter()
        if pytest_filter:
            return self.test_file + ", " + pytest_filter
        return self.test_file

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TestRun):
            return False

        ret = self.test_file == other.test_file
        ret = ret and self._included == other._included
        ret = ret and self._excluded == other._excluded
        return ret

    def __hash__(self) -> int:
        return hash((self.test_file, self._included, self._excluded))

    def __or__(self, other: TestRun) -> TestRun:
        """
        To OR/Union test runs means to run all the tests that either of the two runs specify.
        """

        # Is any file empty?
        if self.is_empty():
            return other
        if other.is_empty():
            return copy(self)

        # If not, ensure we have the same file
        assert (
            self.test_file == other.test_file
        ), f"Can't exclude {other} from {self} because they're not the same test file"

        # 4 possible cases:

        # 1. Either file is the full file, so union is everything
        if self.is_full_file() or other.is_full_file():
            # The union is the whole file
            return TestRun(self.test_file)

        # 2. Both files only run what's in _included, so union is the union of the two sets
        if self._included and other._included:
            return TestRun(
                self.test_file, included=self._included.union(other._included)
            )

        # 3. Both files only exclude what's in _excluded, so union is the intersection of the two sets
        if self._excluded and other._excluded:
            return TestRun(
                self.test_file, excluded=self._excluded.intersection(other._excluded)
            )

        # 4. One file includes and the other excludes, so we then continue excluding the _excluded set minus
        #    whatever is in the _included set
        included = self._included | other._included
        excluded = self._excluded | other._excluded
        return TestRun(self.test_file, excluded=excluded - included)

    def __sub__(self, other: TestRun) -> TestRun:
        """
        To subtract test runs means to run all the tests in the first run except for what the second run specifies.
        """

        # Is any file empty?
        if self.is_empty():
            return TestRun.empty()
        if other.is_empty():
            return copy(self)

        # Are you trying to subtract tests that don't even exist in this test run?
        if self.test_file != other.test_file:
            return copy(self)

        # You're subtracting everything?
        if other.is_full_file():
            return TestRun.empty()

        def return_inclusions_or_empty(inclusions: frozenset[str]) -> TestRun:
            if inclusions:
                return TestRun(self.test_file, included=inclusions)
            return TestRun.empty()

        if other._included:
            if self._included:
                return return_inclusions_or_empty(self._included - other._included)
            else:
                return TestRun(
                    self.test_file, excluded=self._excluded | other._included
                )
        else:
            if self._included:
                return return_inclusions_or_empty(self._included & other._excluded)
            else:
                return return_inclusions_or_empty(other._excluded - self._excluded)

    def __and__(self, other: TestRun) -> TestRun:
        if self.test_file != other.test_file:
            return TestRun.empty()

        return (self | other) - (self - other) - (other - self)

    def to_json(self) -> dict[str, Any]:
        r: dict[str, Any] = {
            "test_file": self.test_file,
        }
        if self._included:
            r["included"] = list(self._included)
        if self._excluded:
            r["excluded"] = list(self._excluded)
        return r

    @staticmethod
    def from_json(json: dict[str, Any]) -> TestRun:
        return TestRun(
            json["test_file"],
            included=json.get("included", []),
            excluded=json.get("excluded", []),
        )


@total_ordering
class ShardedTest:
    test: TestRun
    shard: int
    num_shards: int
    time: float | None  # In seconds

    def __init__(
        self,
        test: TestRun | str,
        shard: int,
        num_shards: int,
        time: float | None = None,
    ) -> None:
        if isinstance(test, str):
            test = TestRun(test)
        self.test = test
        self.shard = shard
        self.num_shards = num_shards
        self.time = time

    @property
    def name(self) -> str:
        return self.test.test_file

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ShardedTest):
            return False
        return (
            self.test == other.test
            and self.shard == other.shard
            and self.num_shards == other.num_shards
            and self.time == other.time
        )

    def __repr__(self) -> str:
        ret = f"{self.test} {self.shard}/{self.num_shards}"
        if self.time:
            ret += f" ({self.time}s)"

        return ret

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, ShardedTest):
            raise NotImplementedError

        # This is how the list was implicity sorted when it was a NamedTuple
        if self.name != other.name:
            return self.name < other.name
        if self.shard != other.shard:
            return self.shard < other.shard
        if self.num_shards != other.num_shards:
            return self.num_shards < other.num_shards

        # None is the smallest value
        if self.time is None:
            return True
        if other.time is None:
            return False
        return self.time < other.time

    def __str__(self) -> str:
        return f"{self.test} {self.shard}/{self.num_shards}"

    def get_time(self, default: float = 0) -> float:
        return self.time if self.time is not None else default

    def get_pytest_args(self) -> list[str]:
        filter = self.test.get_pytest_filter()
        if filter:
            return ["-k", self.test.get_pytest_filter()]
        return []
