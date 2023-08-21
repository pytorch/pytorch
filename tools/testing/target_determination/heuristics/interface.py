from abc import abstractmethod
from typing import Any, Dict, List


class TestPrioritizations:
    """
    Describes the results of whether heuristics consider a test relevant or not.
    Heuristics can leave a list empty if they don't consider any tests relevant.

    Heuristics can leave the unranked_relevance list empty to imply all unmentioned tests are irrelevant

    Important: Lists of tests must always be returned in a deterministic order,
               otherwise it breaks the test sharding logic
    """

    highly_relevant: List[str]
    probably_relevant: List[str]
    # For when we don't know if the test is relevant to the PR or not
    unranked_relevance: List[str]
    # future cateories could include 'definitely not relevant' and 'probably not relevant'

    def __init__(self) -> None:
        self.highly_relevant = []
        self.probably_relevant = []
        self.unranked_relevance = []

    @staticmethod
    def _merge_tests(
        current_tests: List[str], new_tests: List[str], exclude_tests: List[str]
    ) -> List[str]:
        """
        We append all new tests to the current tests, while preserving the sorting on the new_tests
        However, exclude any specified tests which have now moved to a higher priority list
        """
        current_tests.extend(new_tests)
        current_tests = [
            test for test in current_tests if test not in exclude_tests
        ]  # skip the excluded tests
        current_tests = list(dict.fromkeys(current_tests))  # remove dupes

        return current_tests

    def integrate_priorities(self, other: "TestPrioritizations") -> None:
        """
        Integrates priorities from another TestPrioritizations object.

        Assumes tests are only shuffled around beteen the lists, with no tests added or removed.
        """
        # only add new tests to the list, while preserving the sorting
        self.highly_relevant = TestPrioritizations._merge_tests(
            self.highly_relevant, other.highly_relevant, exclude_tests=[]
        )

        self.probably_relevant = TestPrioritizations._merge_tests(
            self.probably_relevant,
            other.probably_relevant,
            exclude_tests=self.highly_relevant,
        )

        # We don't expect every heuristics to list all unranked tests. Easier to compute the list from scratch
        ranked_tests = self.highly_relevant + self.probably_relevant
        self.unranked_relevance = [
            t for t in self.unranked_relevance if t not in ranked_tests
        ]

    def find_conflicts(self) -> List[str]:
        """
        Returns a list of tests which are in multiple lists.
        """
        tests = self.highly_relevant + self.probably_relevant + self.unranked_relevance
        return [test for test in tests if tests.count(test) > 1]

    def print_info(self) -> None:
        def _print_tests(label: str, tests: List[str]) -> None:
            if not tests:
                return

            print(f"{label} tests ({len(tests)}):")
            for test in tests:
                if test in tests:
                    print(f"  {test}")

        _print_tests("Highly relevant", self.highly_relevant)
        _print_tests("Probably relevant", self.probably_relevant)
        _print_tests("Unranked relevance", self.unranked_relevance)

        conflicts = self.find_conflicts()
        if conflicts:
            print(f"WARNING: {len(conflicts)} tests are in multiple lists:")
            for test in conflicts:
                lists = []
                for list_name in ["highly_relevant", "probably_relevant", "unranked"]:
                    if test in getattr(self, list_name):
                        lists.append(list_name)
                print(f"  {test} is in {' ,'.join(lists)}")


class HeuristicInterface:
    """Interface for all heuristics."""

    name: str
    description: str

    @abstractmethod
    def __init__(self, **kwargs: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def get_test_priorities(self, tests: List[str]) -> TestPrioritizations:
        pass

    def __str__(self) -> str:
        # returns the concrete class's name, and optionally any constructor arguments params

        return self.__class__.__name__
