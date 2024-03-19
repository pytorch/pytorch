from abc import abstractmethod
from copy import copy
from typing import Any, Dict, FrozenSet, Iterable, Iterator, List, Tuple

from tools.testing.test_run import TestRun


class TestPrioritizations:
    """
    Describes the results of whether heuristics consider a test relevant or not.

    All the different ranks of tests are disjoint, meaning a test can only be in one category, and they are only
    declared at initialization time.

    A list can be empty if a heuristic doesn't consider any tests to be in that category.

    Important: Lists of tests must always be returned in a deterministic order,
               otherwise it breaks the test sharding logic
    """

    _original_tests: FrozenSet[str]
    _test_scores: Dict[TestRun, float]

    def __init__(
        self,
        tests_being_ranked: Iterable[str],  # The tests that are being prioritized.
        scores: Dict[TestRun, float],
    ) -> None:
        self._original_tests = frozenset(tests_being_ranked)
        self._test_scores = {TestRun(test): 0.0 for test in self._original_tests}

        for test, score in scores.items():
            self.set_test_score(test, score)

        self.validate()

    def validate(self) -> None:
        # Union all TestRuns that contain include/exclude pairs
        all_tests = self._test_scores.keys()
        files = {}
        for test in all_tests:
            if test.test_file not in files:
                files[test.test_file] = copy(test)
            else:
                assert (
                    files[test.test_file] & test
                ).is_empty(), (
                    f"Test run `{test}` overlaps with `{files[test.test_file]}`"
                )
                files[test.test_file] |= test

        for test in files.values():
            assert (
                test.is_full_file()
            ), f"All includes should have been excluded elsewhere, and vice versa. Test run `{test}` violates that"

        # Ensure that the set of tests in the TestPrioritizations is identical to the set of tests passed in
        assert self._original_tests == set(
            files.keys()
        ), "The set of tests in the TestPrioritizations must be identical to the set of tests passed in"

    def _traverse_scores(self) -> Iterator[Tuple[float, TestRun]]:
        # Sort by score, then alphabetically by test name
        for test, score in sorted(
            self._test_scores.items(), key=lambda x: (-x[1], str(x[0]))
        ):
            yield score, test

    def set_test_score(self, test_run: TestRun, new_score: float) -> None:
        if test_run.test_file not in self._original_tests:
            return  # We don't need this test

        relevant_test_runs: List[TestRun] = [
            tr for tr in self._test_scores.keys() if tr & test_run and tr != test_run
        ]

        # Set the score of all the tests that are covered by test_run to the same score
        self._test_scores[test_run] = new_score
        # Set the score of all the tests that are not covered by test_run to original score
        for relevant_test_run in relevant_test_runs:
            old_score = self._test_scores[relevant_test_run]
            del self._test_scores[relevant_test_run]

            not_to_be_updated = relevant_test_run - test_run
            if not not_to_be_updated.is_empty():
                self._test_scores[not_to_be_updated] = old_score
        self.validate()

    def add_test_score(self, test_run: TestRun, score_to_add: float) -> None:
        if test_run.test_file not in self._original_tests:
            return

        relevant_test_runs: List[TestRun] = [
            tr for tr in self._test_scores.keys() if tr & test_run
        ]

        for relevant_test_run in relevant_test_runs:
            old_score = self._test_scores[relevant_test_run]
            del self._test_scores[relevant_test_run]

            intersection = relevant_test_run & test_run
            if not intersection.is_empty():
                self._test_scores[intersection] = old_score + score_to_add

            not_to_be_updated = relevant_test_run - test_run
            if not not_to_be_updated.is_empty():
                self._test_scores[not_to_be_updated] = old_score

        self.validate()

    def get_all_tests(self) -> List[TestRun]:
        """Returns all tests in the TestPrioritizations"""
        return [x[1] for x in self._traverse_scores()]

    def get_top_per_tests(self, n: int) -> Tuple[List[TestRun], List[TestRun]]:
        """Divides list of tests into two based on the top n% of scores.  The
        first list is the top, and the second is the rest."""
        tests = [x[1] for x in self._traverse_scores()]
        return tests[: n * len(tests) // 100], tests[n * len(tests) // 100 :]

    def get_info_str(self, verbose: bool = True) -> str:
        info = ""

        for score, test in self._traverse_scores():
            if not verbose and score == 0:
                continue
            info += f"  {test} ({score})\n"

        return info.rstrip()

    def print_info(self) -> None:
        print(self.get_info_str())

    def get_priority_info_for_test(self, test_run: TestRun) -> Dict[str, Any]:
        """Given a failing test, returns information about it's prioritization that we want to emit in our metrics."""
        for idx, (score, test) in enumerate(self._traverse_scores()):
            #  Different heuristics may result in a given test file being split
            #  into different test runs, so look for the overlapping tests to
            #  find the match
            if test & test_run:
                return {"position": idx, "score": score}
        raise AssertionError(f"Test run {test_run} not found")

    def get_test_stats(self, test: TestRun) -> Dict[str, Any]:
        return {
            "test_name": test.test_file,
            "test_filters": test.get_pytest_filter(),
            **self.get_priority_info_for_test(test),
            "max_score": max(score for score, _ in self._traverse_scores()),
            "min_score": min(score for score, _ in self._traverse_scores()),
            "all_scores": {
                str(test): score for test, score in self._test_scores.items()
            },
        }

    def to_json(self) -> Dict[str, Any]:
        """
        Returns a JSON dict that describes this TestPrioritizations object.
        """
        json_dict = {
            "_test_scores": [
                (test.to_json(), score)
                for test, score in self._test_scores.items()
                if score != 0
            ],
            "_original_tests": list(self._original_tests),
        }
        return json_dict

    @staticmethod
    def from_json(json_dict: Dict[str, Any]) -> "TestPrioritizations":
        """
        Returns a TestPrioritizations object from a JSON dict.
        """
        test_prioritizations = TestPrioritizations(
            tests_being_ranked=json_dict["_original_tests"],
            scores={
                TestRun.from_json(testrun_json): score
                for testrun_json, score in json_dict["_test_scores"]
            },
        )
        return test_prioritizations

    def amend_tests(self, tests: List[str]) -> None:
        """
        Removes tests that are not in the given list from the
        TestPrioritizations.  Adds tests that are in the list but not in the
        TestPrioritizations.
        """
        valid_scores = {
            test: score
            for test, score in self._test_scores.items()
            if test.test_file in tests
        }
        self._test_scores = valid_scores

        for test in tests:
            if test not in self._original_tests:
                self._test_scores[TestRun(test)] = 0
        self._original_tests = frozenset(tests)

        self.validate()


class AggregatedHeuristics:
    """
    Aggregates the results across all heuristics.

    It saves the individual results from each heuristic and exposes an aggregated view.
    """

    _heuristic_results: Dict[
        "HeuristicInterface", TestPrioritizations
    ]  # Key is the Heuristic's name. Dicts will preserve the order of insertion, which is important for sharding

    _all_tests: FrozenSet[str]

    def __init__(self, all_tests: List[str]) -> None:
        self._all_tests = frozenset(all_tests)
        self._heuristic_results = {}
        self.validate()

    def validate(self) -> None:
        for heuristic, heuristic_results in self._heuristic_results.items():
            heuristic_results.validate()
            assert (
                heuristic_results._original_tests == self._all_tests
            ), f"Tests in {heuristic.name} are not the same as the tests in the AggregatedHeuristics"

    def add_heuristic_results(
        self, heuristic: "HeuristicInterface", heuristic_results: TestPrioritizations
    ) -> None:
        if heuristic in self._heuristic_results:
            raise ValueError(f"We already have heuristics for {heuristic.name}")

        self._heuristic_results[heuristic] = heuristic_results
        self.validate()

    def get_aggregated_priorities(
        self, include_trial: bool = False
    ) -> TestPrioritizations:
        """
        Returns the aggregated priorities across all heuristics.
        """
        valid_heuristics = {
            heuristic: heuristic_results
            for heuristic, heuristic_results in self._heuristic_results.items()
            if not heuristic.trial_mode or include_trial
        }

        new_tp = TestPrioritizations(self._all_tests, {})

        for heuristic_results in valid_heuristics.values():
            for score, testrun in heuristic_results._traverse_scores():
                new_tp.add_test_score(testrun, score)
        new_tp.validate()
        return new_tp

    def get_test_stats(self, test: TestRun) -> Dict[str, Any]:
        """
        Returns the aggregated statistics for a given test.
        """
        stats: Dict[str, Any] = {
            "test_name": test.test_file,
            "test_filters": test.get_pytest_filter(),
        }

        # Get metrics about the heuristics used
        heuristics = []

        for heuristic, heuristic_results in self._heuristic_results.items():
            metrics = heuristic_results.get_priority_info_for_test(test)
            metrics["heuristic_name"] = heuristic.name
            metrics["trial_mode"] = heuristic.trial_mode
            heuristics.append(metrics)

        stats["heuristics"] = heuristics

        stats[
            "aggregated"
        ] = self.get_aggregated_priorities().get_priority_info_for_test(test)

        stats["aggregated_trial"] = self.get_aggregated_priorities(
            include_trial=True
        ).get_priority_info_for_test(test)

        return stats

    def to_json(self) -> Dict[str, Any]:
        """
        Returns a JSON dict that describes this AggregatedHeuristics object.
        """
        json_dict: Dict[str, Any] = {}
        for heuristic, heuristic_results in self._heuristic_results.items():
            json_dict[heuristic.name] = heuristic_results.to_json()

        return json_dict


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

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def __str__(self) -> str:
        return self.name

    @abstractmethod
    def get_prediction_confidence(self, tests: List[str]) -> TestPrioritizations:
        """
        Returns a float ranking ranging from -1 to 1, where negative means skip,
        positive means run, 0 means no idea, and magnitude = how confident the
        heuristic is. Used by AggregatedHeuristicsRankings.
        """
        pass
