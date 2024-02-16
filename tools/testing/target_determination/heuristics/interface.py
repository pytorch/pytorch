import operator
import sys
from abc import abstractmethod
from copy import copy
from enum import Enum
from functools import total_ordering
from itertools import chain
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
)

from tools.testing.test_run import TestRun, TestRuns


# Note: Keep the implementation of Relevance private to this file so
# that it's easy to change in the future as we discover what's needed
@total_ordering
class Relevance(Enum):
    HIGH = 4
    PROBABLE = 3
    UNRANKED = 2
    UNLIKELY = 1
    NONE = 0

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Relevance):
            return False

        return self.value == other.value

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Relevance):
            raise NotImplementedError(f"Can't compare {self} to {other}")

        return self.value < other.value

    def __hash__(self) -> int:
        return self.value

    @staticmethod
    def priority_traversal() -> Iterator["Relevance"]:
        yield Relevance.HIGH
        yield Relevance.PROBABLE
        yield Relevance.UNRANKED
        yield Relevance.UNLIKELY
        yield Relevance.NONE


METRIC_RELEVANCE_GROUP = "relevance_group"
METRIC_ORDER_WITHIN_RELEVANCE_GROUP = "order_within_relevance_group"
METRIC_NUM_TESTS_IN_RELEVANCE_GROUP = "num_tests_in_relevance_group"
METRIC_ORDER_OVERALL = "order_overall"
METRIC_HEURISTIC_NAME = "heuristic_name"


class TestPrioritizations:
    """
    Describes the results of whether heuristics consider a test relevant or not.

    All the different ranks of tests are disjoint, meaning a test can only be in one category, and they are only
    declared at initialization time.

    A list can be empty if a heuristic doesn't consider any tests to be in that category.

    Important: Lists of tests must always be returned in a deterministic order,
               otherwise it breaks the test sharding logic
    """

    _test_priorities: List[List[TestRun]]
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

        self._test_priorities = [[] for _ in range(5)]
        # Setup the initial priorities
        self._test_priorities[Relevance.UNRANKED.value] = [
            TestRun(test) for test in tests_being_ranked
        ]

        for test in high_relevance or []:
            self.set_test_relevance(TestRun(test), Relevance.HIGH)
        for test in probable_relevance or []:
            self.set_test_relevance(TestRun(test), Relevance.PROBABLE)
        for test in unranked_relevance or []:
            self.set_test_relevance(TestRun(test), Relevance.UNRANKED)
        for test in unlikely_relevance or []:
            self.set_test_relevance(TestRun(test), Relevance.UNLIKELY)
        for test in no_relevance or []:
            self.set_test_relevance(TestRun(test), Relevance.NONE)

        self.validate_test_priorities()

    def _traverse_priorities(self) -> Iterator[Tuple[Relevance, List[TestRun]]]:
        for relevance in Relevance.priority_traversal():
            yield (relevance, self._test_priorities[relevance.value])

    def get_pointer_to_test(self, test_run: TestRun) -> Iterator[Tuple[Relevance, int]]:
        """
        Returns all test runs that contain any subset of the given test_run and their current relevance.

        self._test_priorities should NOT have items added or removed form it while iterating over the
        results of this function.
        """
        # Find a test run that contains the given TestRun and it's relevance.
        found_match = False
        for relevance, tests in self._traverse_priorities():
            for idx, existing_test_run in enumerate(tests):
                # Does the existing test run contain any of the test we're looking for?
                shared_test = existing_test_run & test_run
                if not shared_test.is_empty():
                    found_match = True
                    yield (Relevance(relevance), idx)

        if not found_match:
            raise ValueError(f"Test {test_run} not found in any relevance group")

    def _update_test_relevance(
        self,
        test_run: TestRun,
        new_relevance: Relevance,
        acceptable_relevance_fn: Callable[[Relevance, Relevance], bool],
    ) -> None:
        """
        Updates the test run's relevance to the new relevance.

        If the tests in the test run were previously split up into multiple test runs, all the chunks at a lower
        relevance will be merged into one new test run at the new relevance, appended to the end of the relevance group.

        However, any tests in a test run that are already at the desired relevance will be left alone, keeping it's
        original place in the relevance group.
        """
        if test_run.test_file not in self._original_tests:
            return  # We don't need this test

        # The tests covered by test_run could potentially have been split up into
        # multiple test runs, each at a different relevance. Let's make sure to bring
        # all of them up to the minimum relevance
        upgraded_tests = TestRun.empty()
        tests_to_remove = []
        for curr_relevance, test_run_idx in self.get_pointer_to_test(test_run):
            if acceptable_relevance_fn(curr_relevance, new_relevance):
                # This test is already at the desired relevance
                continue  # no changes needed

            test_run_to_rerank = self._test_priorities[curr_relevance.value][
                test_run_idx
            ]
            # Remove the requested tests from their current relevance group, to be added to the new one
            remaining_tests = test_run_to_rerank - test_run
            upgraded_tests |= test_run_to_rerank & test_run

            # Remove the tests that are being upgraded
            if remaining_tests:
                self._test_priorities[curr_relevance.value][
                    test_run_idx
                ] = remaining_tests
            else:
                # List traversal prevents us from deleting these immediately, so note them for later
                tests_to_remove.append((curr_relevance, test_run_idx))

        for relevance, test_idx in tests_to_remove:
            del self._test_priorities[relevance.value][test_idx]

        # And add them to the desired relevance group
        if upgraded_tests:
            self._test_priorities[new_relevance.value].append(upgraded_tests)

    def set_test_relevance(self, test_run: TestRun, new_relevance: Relevance) -> None:
        return self._update_test_relevance(test_run, new_relevance, operator.eq)

    def raise_test_relevance(self, test_run: TestRun, new_relevance: Relevance) -> None:
        return self._update_test_relevance(test_run, new_relevance, operator.ge)

    def validate_test_priorities(self) -> None:
        # Union all TestRuns that contain include/exclude pairs
        all_tests = self.get_all_tests()
        files = {}
        for test in all_tests:
            if test.test_file not in files:
                files[test.test_file] = copy(test)
            else:
                files[test.test_file] |= test

        for test in files.values():
            assert (
                test.is_full_file()
            ), f"All includes should have been excluded elsewhere, and vice versa. Test run `{test}` violates that"

        # Ensure that the set of tests in the TestPrioritizations is identical to the set of tests passed in
        assert self._original_tests == set(
            files.keys()
        ), "The set of tests in the TestPrioritizations must be identical to the set of tests passed in"

    def integrate_priorities(self, other: "TestPrioritizations") -> None:
        """
        Integrates priorities from another TestPrioritizations object.

        The final result takes all tests from the `self` and rearranges them based on priorities from `other`.
        Currently it will only raise the priority of a test, never lower it.
        """
        assert (
            self._original_tests == other._original_tests
        ), "Both tests should stem from the same original test list"

        for relevance, _ in other._traverse_priorities():
            if relevance > Relevance.UNRANKED:
                for test in other._test_priorities[relevance.value]:
                    self.raise_test_relevance(test, relevance)

        self.validate_test_priorities()
        return

    def get_all_tests(self) -> TestRuns:
        """Returns all tests in the TestPrioritizations"""
        return tuple(chain(*self._test_priorities))

    def get_prioritized_tests(self) -> TestRuns:
        return self.get_high_relevance_tests() + self.get_probable_relevance_tests()

    def get_high_relevance_tests(self) -> TestRuns:
        return tuple(test for test in self._test_priorities[Relevance.HIGH.value])

    def get_probable_relevance_tests(self) -> TestRuns:
        return tuple(test for test in self._test_priorities[Relevance.PROBABLE.value])

    def get_unranked_relevance_tests(self) -> TestRuns:
        return tuple(test for test in self._test_priorities[Relevance.UNRANKED.value])

    def get_unlikely_relevance_tests(self) -> TestRuns:
        return tuple(test for test in self._test_priorities[Relevance.UNLIKELY.value])

    def get_none_relevance_tests(self) -> TestRuns:
        return tuple(test for test in self._test_priorities[Relevance.NONE.value])

    def get_info_str(self) -> str:
        info = ""

        def _test_info(label: str, tests: List[TestRun]) -> str:
            if not tests:
                return ""

            s = f"{label} tests ({len(tests)}):\n"
            for test in tests:
                if test in tests:
                    s += f"  {test}\n"
            return s

        for relevance_group, tests in self._traverse_priorities():
            if relevance_group == Relevance.UNRANKED:
                continue
            info += _test_info(
                f"{Relevance(relevance_group).name.title()} Relevance", tests
            )

        return info.strip()

    def print_info(self) -> None:
        print(self.get_info_str())

    def _get_test_relevance_group(self, test_run: TestRun) -> Relevance:
        """
        Returns the relevance of the given test run.
        If the heuristic split this test run among multiple runs, then return the
        highest relevance of any of the test runs.
        """
        for relevance_group, tests in self._traverse_priorities():
            #  Different heuristics may result in a given test file being split
            #  into different test runs, so look for the overlapping tests to
            #  find the match
            if any(t & test_run for t in tests):
                return Relevance(relevance_group)

        raise ValueError(f"Test {test_run} not found in any relevance group")

    def _get_test_order(self, test_run: TestRun) -> int:
        """
        Returns the rank this heuristic suggested for the test run.
        If the heuristic split this test run among multiple runs, then return the
        highest relevance of any of the test runs.
        """
        base_rank = 0

        for _, relevance_group_tests in self._traverse_priorities():
            for idx, test in enumerate(relevance_group_tests):
                #  Different heuristics may result in a given test file being split
                #  into different test runs, so look for the overlapping tests to
                #  find the match
                if test & test_run:
                    return base_rank + idx
            base_rank += len(relevance_group_tests)

        raise ValueError(f"Test {test_run} not found in any relevance group")

    def _get_test_order_within_relevance_group(self, test_run: TestRun) -> int:
        """
        Returns the highest test order of any test class within the same relevance group.
        If the heuristic split this test run among multiple runs, then return the
        highest relevance of any of the test runs.
        """
        for _, relevance_group_tests in self._traverse_priorities():
            for idx, test in enumerate(relevance_group_tests):
                #  Different heuristics may result in a given test file being split
                #  into different test runs, so look for the overlapping tests to
                #  find the match
                if test & test_run:
                    return idx

        raise ValueError(f"Test {test_run} not found in any relevance group")

    def get_priority_info_for_test(self, test_run: TestRun) -> Dict[str, Any]:
        """Given a failing test, returns information about it's prioritization that we want to emit in our metrics."""
        relevance = self._get_test_relevance_group(test_run)
        return {
            METRIC_RELEVANCE_GROUP: relevance.name,
            METRIC_ORDER_WITHIN_RELEVANCE_GROUP: self._get_test_order_within_relevance_group(
                test_run
            ),
            METRIC_NUM_TESTS_IN_RELEVANCE_GROUP: len(
                self._test_priorities[relevance.value]
            ),
            METRIC_ORDER_OVERALL: self._get_test_order(test_run),
        }


class AggregatedHeuristics:
    """
    Aggregates the results across all heuristics.

    It saves the individual results from each heuristic and exposes an aggregated view.
    """

    _heuristic_results: Dict[
        "HeuristicInterface", TestPrioritizations
    ]  # Key is the Heuristic's name. Dicts will preserve the order of insertion, which is important for sharding

    unranked_tests: Tuple[str, ...]

    def __init__(self, unranked_tests: List[str]) -> None:
        self.unranked_tests = tuple(unranked_tests)
        self._heuristic_results = {}

    def add_heuristic_results(
        self, heuristic: "HeuristicInterface", heuristic_results: TestPrioritizations
    ) -> None:
        if heuristic in self._heuristic_results:
            raise ValueError(f"We already have heuristics for {heuristic.name}")

        self._heuristic_results[heuristic] = heuristic_results

    def get_aggregated_priorities(
        self, include_trial: bool = False
    ) -> TestPrioritizations:
        """
        Returns the aggregated priorities across all heuristics.
        """
        valid_heuristics = {
            heuristic: heuristic_results
            for heuristic, heuristic_results in self._heuristic_results.items()
            if heuristic.trial_mode == include_trial
        }

        def get_left_right_intersection(
            left_testrun: TestRun, right_testrun: TestRun
        ) -> Tuple[TestRun, TestRun, TestRun]:
            only_left = left_testrun - right_testrun
            only_right = right_testrun - left_testrun
            both = left_testrun & right_testrun
            return only_left, only_right, both

        how_many_probable_is_equal_to_high = 3
        relevance_value = {
            Relevance.HIGH: 1,
            Relevance.PROBABLE: 1 / how_many_probable_is_equal_to_high,
            Relevance.UNRANKED: 0,
            Relevance.UNLIKELY: -1 / how_many_probable_is_equal_to_high,
            Relevance.NONE: -1,
        }

        # Get an ordering of the tests based on insertion order, which is
        # preserved when sorting
        ordering: List[Tuple[float, TestRun]] = []
        for heuristic_results in valid_heuristics.values():
            for relevance, testruns in heuristic_results._traverse_priorities():
                if relevance == Relevance.UNRANKED:
                    continue
                for testrun in testruns:
                    curr = testrun
                    new_ordering = []
                    for existing_relevance, existing_testrun in ordering:
                        if curr.test_file == existing_testrun.test_file:
                            left, right, intersection = get_left_right_intersection(
                                existing_testrun, curr
                            )
                            if not intersection.is_empty():
                                new_ordering.append(
                                    (
                                        existing_relevance + relevance_value[relevance],
                                        intersection,
                                    )
                                )
                            if not left.is_empty():
                                new_ordering.append((existing_relevance, left))
                            curr = right
                        else:
                            new_ordering.append((existing_relevance, existing_testrun))
                    if not curr.is_empty():
                        new_ordering.append((relevance_value[relevance], curr))
                    ordering = new_ordering

        aggregated_priorities = TestPrioritizations(
            tests_being_ranked=self.unranked_tests,
        )

        def get_relevance(val: float) -> Relevance:
            if val >= 1:
                return Relevance.HIGH
            if 0 < val < 1:
                return Relevance.PROBABLE
            if val == 0:
                return Relevance.UNRANKED
            if -1 < val < 0:
                return Relevance.UNLIKELY
            # val <= -1:
            return Relevance.NONE

        # Sort by relevance value (ex test that is marked high relevance by two
        # heuristics has value 2 and should come before test that was only
        # marked as high relevance by one heuristic), and then group by value
        for relevance_val, testrun in sorted(
            ordering, key=lambda x: x[0], reverse=True
        ):
            aggregated_priorities.set_test_relevance(
                testrun, get_relevance(relevance_val)
            )

        aggregated_priorities.validate_test_priorities()

        return aggregated_priorities

    def get_test_stats(self, test: TestRun) -> Dict[str, Any]:
        """
        Returns the aggregated statistics for a given test.
        """
        stats: Dict[str, Any] = {
            "test_name": test.test_file,
            "test_filters": test.get_pytest_filter(),
        }

        # Get baseline metrics assuming we didn't have any TD heuristics
        baseline_priorities = TestPrioritizations(
            tests_being_ranked=self.unranked_tests
        )
        baseline_stats = baseline_priorities.get_priority_info_for_test(test)
        baseline_stats["heuristic_name"] = "baseline"
        stats["without_heuristics"] = baseline_stats

        # Get metrics about the heuristics used
        heuristics = []

        # Figure out which heuristic gave this test the highest priority (if any)
        highest_ranking_heuristic = None
        highest_ranking_heuristic_order: int = sys.maxsize

        # And figure out how many heuristics suggested prioritizing this test
        num_heuristics_prioritized_by = 0

        for heuristic, heuristic_results in self._heuristic_results.items():
            metrics = heuristic_results.get_priority_info_for_test(test)
            metrics["heuristic_name"] = heuristic.name
            metrics["trial_mode"] = heuristic.trial_mode
            heuristics.append(metrics)

            if not heuristic.trial_mode and heuristic_results._get_test_relevance_group(
                test
            ) in [
                Relevance.HIGH,
                Relevance.PROBABLE,
            ]:
                num_heuristics_prioritized_by += 1

                # "highest_ranking_heuristic" should only consider heuristics that actually prioritize the test.
                # Sometimes an UNRANKED heuristic could be have an overall order above a PRIORITIZED heuristic
                # because it was randomly sorted higher initially, while the heuristic that actually prioritized it
                # used other data to determined it to be slighlty less relevant than other tests.
                if metrics[METRIC_ORDER_OVERALL] < highest_ranking_heuristic_order:
                    highest_ranking_heuristic = heuristic.name
                    highest_ranking_heuristic_order = metrics[METRIC_ORDER_OVERALL]

        stats["heuristics"] = heuristics

        # Easier to compute here than in rockset
        stats["num_heuristics_prioritized_by"] = num_heuristics_prioritized_by

        stats[
            "aggregated"
        ] = self.get_aggregated_priorities().get_priority_info_for_test(test)

        stats["aggregated_trial"] = self.get_aggregated_priorities(
            include_trial=True
        ).get_priority_info_for_test(test)

        if highest_ranking_heuristic:
            stats["highest_ranking_heuristic"] = highest_ranking_heuristic

        return stats


class HeuristicInterface:
    """
    Interface for all heuristics.
    """

    description: str

    # When trial mode is set to True, this heuristic's predictions will not be used
    # to reorder tests. It's results will however be emitted in the metrics.
    trial_mode: bool

    @abstractmethod
    def __init__(self, **kwargs: Any) -> None:
        self.trial_mode = kwargs.get("trial_mode", False)  # type: ignore[assignment]

    @abstractmethod
    def get_test_priorities(self, tests: List[str]) -> TestPrioritizations:
        """
        Returns the prioritizations for the given tests.

        The set of test in TestPrioritizations _must_ be identical to the set of tests passed in.
        """
        pass

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def __str__(self) -> str:
        return self.name

    @abstractmethod
    def get_prediction_confidence(self, tests: List[str]) -> Dict[str, float]:
        """
        Like get_test_priorities, but instead returns a float ranking ranging
        from -1 to 1, where negative means skip, positive means run, 0 means no
        idea, and magnitude = how confident the heuristic is. Used by
        AggregatedHeuristicsRankings.
        """
        pass
