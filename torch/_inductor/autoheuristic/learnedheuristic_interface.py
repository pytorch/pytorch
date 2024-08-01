from typing import List, Optional

from torch._inductor.autoheuristic.autoheuristic_utils import (
    AHContext,
    AHMetadata,
    Choice,
)


class LearnedHeuristic:
    """
    LearnedHeuristic is a base class for all learned heuristics.
    """

    def __init__(self) -> None:
        pass

    def check_precondition(
        self,
        metadata: AHMetadata,
        context: AHContext,
    ) -> bool:
        return True

    def get_decision(
        self, context: AHContext, choices: List[Choice]
    ) -> Optional[Choice]:
        choice2feedback = {}
        for choice in choices:
            predicted_feedback = self.get_feedback(context, choice)
            choice2feedback[choice] = predicted_feedback
        sorted_choices_feedback = sorted(choice2feedback.items(), key=lambda t: t[1])
        highest_feedback = sorted_choices_feedback[-1][1]
        second_highest_feedback = sorted_choices_feedback[-2][1]
        if highest_feedback / second_highest_feedback > self.get_speedup_threshold():
            return sorted_choices_feedback[-1][0]
        # We are not sure which choice is the best one
        return None

    def get_feedback(self, context: AHContext, choice: Choice) -> float:
        return 1.0

    def get_speedup_threshold(self) -> float:
        return 1.0

    def get_name(self) -> str:
        return ""
