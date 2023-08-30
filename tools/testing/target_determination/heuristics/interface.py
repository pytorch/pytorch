from abc import abstractmethod
from enum import Enum
from itertools import chain
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Tuple

StringTuple = Tuple[str, ...]


# Note: Keep the implementation of Relevance private to this file so
# that it's easy to change in the future as we discover what's needed
class Relevance(Enum):
    HIGH = 0
    PROBABLE = 1
    UNRANKED = 2
    UNLIKELY = 3  # Not yet supported. Needs more infra to be usable
    NONE = 4  # Not yet supported. Needs more infra to be usable


class TestPrioritizations:
    """
    Describes the results of whether heuristics consider a test relevant or not.

    All the different ranks of tests are disjoint, meaning a test can only be in one category, and they are only
    declared at initization time.

    A list can be empty if a heuristic doesn't consider any tests to be in that category.

    Important: Lists of tests must always be returned in a deterministic order,
               otherwise it breaks the test sharding logic
    """

    _test_priorities: List[StringTuple]  # This list MUST be ordered by Relevance
    _original_tests: FrozenSet[str]

    def __init__(
        self,
        tests_being_ranked: Iterable[str],  # The tests that are being prioritized.
        high_relevance: Optional[List[str]] = None,
        probable_relevance: Optional[List[str]] = None,
        unranked_relevance: Optional[List[str]] = None,
        unlikely_relevance: Optional[List[str]] = None,
        no_relevance: Optional[List[str]] = None,
    ) -> None:
        self._original_tests = frozenset(tests_being_ranked)

        self._test_priorities = [tuple() for _ in range(5)]

        self._test_priorities[Relevance.HIGH.value] = self.filter_out_extra_tests(
            high_relevance
        )
        self._test_priorities[Relevance.PROBABLE.value] = self.filter_out_extra_tests(
            probable_relevance
        )
        self._test_priorities[Relevance.UNRANKED.value] = self.filter_out_extra_tests(
            unranked_relevance
        )
        self._test_priorities[Relevance.UNLIKELY.value] = self.filter_out_extra_tests(
            unlikely_relevance
        )
        self._test_priorities[Relevance.NONE.value] = self.filter_out_extra_tests(
            no_relevance
        )

        # If any of the original tests were missed from the other lists, add them to the unranked_relevance list
        missing_tests = sorted(self._original_tests - set(self.get_all_tests()))
        self._test_priorities[Relevance.UNRANKED.value] = self._test_priorities[
            Relevance.UNRANKED.value
        ] + tuple(missing_tests)

        self.validate_test_priorities()

    def filter_out_extra_tests(
        self, relevance_group: Optional[List[str]]
    ) -> StringTuple:
        if not relevance_group:
            return tuple()
        return tuple(filter(lambda test: test in self._original_tests, relevance_group))

    def validate_test_priorities(self) -> None:
        # Ensure that the set of tests in the TestPrioritizations is identical to the set of tests passed in
        assert self._original_tests == set(
            self.get_all_tests()
        ), "The set of tests in the TestPrioritizations must be identical to the set of tests passed in"

    @staticmethod
    def _merge_tests(
        current_tests: Iterable[str],
        new_tests: Iterable[str],
        higher_pri_tests: Iterable[str],
    ) -> StringTuple:
        """
        We append all new tests to the current tests, while preserving the sorting on the new_tests
        However, exclude any specified tests which have now moved to a higher priority list or tests
        that weren't originally in the self's TestPrioritizations
        """
        merged_tests = [
            test
            for test in chain(current_tests, new_tests)
            if test not in higher_pri_tests
        ]  # skip the excluded tests
        return tuple(dict.fromkeys(merged_tests))  # remove dupes while preseving order

    def integrate_priorities(self, other: "TestPrioritizations") -> None:
        """
        Integrates priorities from another TestPrioritizations object.

        The final result takes all tests from the `self` and rearranges them based on priorities from `other`.
        If there are tests mentioned in `other` which are not in `self`, those tests are ignored.
        (For example, that can happen if a heuristic reports tests that are not run in the current job)
        """
        assert (
            self._original_tests == other._original_tests
        ), "Both tests should stem from the same original test list"

        higher_pri_tests: List[str] = []
        for relevance, _ in enumerate(self._test_priorities):
            self._test_priorities[relevance] = TestPrioritizations._merge_tests(
                current_tests=self._test_priorities[relevance],
                new_tests=other._test_priorities[relevance],
                higher_pri_tests=higher_pri_tests,
            )

            # Don't let the tests we just added to the current relevance group be added to a lower relevance group
            higher_pri_tests.extend(self._test_priorities[relevance])

        self.validate_test_priorities()

    def get_all_tests(self) -> StringTuple:
        """Returns all tests in the TestPrioritizations"""
        return tuple(test for test in chain(*self._test_priorities))

    def get_prioritized_tests(self) -> StringTuple:
        return self.get_high_relevance_tests() + self.get_probable_relevance_tests()

    def get_high_relevance_tests(self) -> StringTuple:
        return tuple(test for test in self._test_priorities[Relevance.HIGH.value])

    def get_probable_relevance_tests(self) -> StringTuple:
        return tuple(test for test in self._test_priorities[Relevance.PROBABLE.value])

    def get_unranked_relevance_tests(self) -> StringTuple:
        return tuple(test for test in self._test_priorities[Relevance.UNRANKED.value])

    def print_info(self) -> None:
        def _print_tests(label: str, tests: StringTuple) -> None:
            if not tests:
                return

            print(f"{label} tests ({len(tests)}):")
            for test in tests:
                if test in tests:
                    print(f"  {test}")

        for relevance_group, tests in enumerate(self._test_priorities):
            _print_tests(f"{Relevance(relevance_group).name.title()} Relevance", tests)


class HeuristicInterface:
    """
    Interface for all heuristics.
    """

    name: str
    description: str

    @abstractmethod
    def __init__(self, **kwargs: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def get_test_priorities(self, tests: List[str]) -> TestPrioritizations:
        """
        Returns the prioritizations for the given tests.

        The set of test in TestPrioritizations _must_ be identical to the set of tests passed in.
        """
        pass

    def __str__(self) -> str:
        return self.__class__.__name__
