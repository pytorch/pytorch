from typing import List, Optional, Tuple

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
        return None

    def get_confidence_threshold(self) -> float:
        return 1.0

    def get_name(self) -> str:
        return ""


class LearnedHeuristicRegression(LearnedHeuristic):
    def __init__(self) -> None:
        super().__init__()

    def get_feedback(self, context: AHContext, choice: Choice) -> float:
        return 1.0

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
        if highest_feedback / second_highest_feedback > self.get_confidence_threshold():
            return sorted_choices_feedback[-1][0]
        # We are not sure which choice is the best one
        return None


class LearnedHeuristicDecision(LearnedHeuristic):
    def __init__(self) -> None:
        super().__init__()

    def get_choice(self, idx: int) -> Optional[str]:
        return None

    def get_decision(
        self, context: AHContext, choices: List[Choice]
    ) -> Optional[Choice]:
        best_choices = self.get_best_choices(context)
        if not best_choices:
            return None
        (best_choice_proba, best_choice_idx) = best_choices[0]
        if best_choice_proba <= self.get_confidence_threshold():
            return None
        return self.get_choice(best_choice_idx)

    def get_best_choices(self, context: AHContext) -> Optional[List[Tuple[float, int]]]:
        return []
