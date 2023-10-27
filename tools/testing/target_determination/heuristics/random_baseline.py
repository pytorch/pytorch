import random
from typing import Any, List

from tools.testing.target_determination.heuristics.interface import (
    HeuristicInterface,
    TestPrioritizations,
)


# This heuristic should never go into production. It is only used for benchmarking our current heuristics.
# It randomly assigns tests to the different categories to show the usefulness of our actual heuristics.
# We should in theory tweak the probabilities to reflect our TTS goal. (Currently 50% of tests should be unranked)
class RandomBaseline(HeuristicInterface):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        assert self.trial_mode, "RandomBaseline should only be used in trial mode"

    def get_test_priorities(self, tests: List[str]) -> TestPrioritizations:
        # Probabilities for each type of test.
        # The sum of all probabilities should be 1.0.
        PROB_UNRANKED = 0.5

        PROB_HIGH_RELEVANCE = 0.25
        PROB_PROBABLE_RELEVANCE = 0.25

        # These catagorizations are not supported yet.
        PROB_NO_RELEVANCE = 0
        PROB_UNLIKELY_RELEVANCE = 0

        assert (
            PROB_UNRANKED
            + PROB_NO_RELEVANCE
            + PROB_UNLIKELY_RELEVANCE
            + PROB_HIGH_RELEVANCE
            + PROB_PROBABLE_RELEVANCE
            == 1.0
        )

        unranked_tests = []
        high_relevance_tests = []
        unlikely_relevance_tests = []
        no_relevance_tests = []
        probable_relevance_tests = []
        for test in tests:
            num = random.random()
            if num <= PROB_UNRANKED:
                unranked_tests.append(test)
            elif num <= PROB_UNRANKED + PROB_HIGH_RELEVANCE:
                high_relevance_tests.append(test)
            elif num <= PROB_UNRANKED + PROB_HIGH_RELEVANCE + PROB_UNLIKELY_RELEVANCE:
                unlikely_relevance_tests.append(test)
            elif (
                num
                <= PROB_UNRANKED
                + PROB_HIGH_RELEVANCE
                + PROB_UNLIKELY_RELEVANCE
                + PROB_NO_RELEVANCE
            ):
                no_relevance_tests.append(test)
            else:
                probable_relevance_tests.append(test)

        test_rankings = TestPrioritizations(
            tests_being_ranked=tests,
            unranked_relevance=unranked_tests,
            high_relevance=high_relevance_tests,
            unlikely_relevance=unlikely_relevance_tests,
            no_relevance=no_relevance_tests,
            probable_relevance=probable_relevance_tests,
        )

        return test_rankings
