import operator

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

    def get_decision(self, context: AHContext, choices: list[Choice]) -> Choice | None:
        return None

    def get_confidence_threshold(self) -> float:
        return 1.0

    def get_name(self) -> str:
        return ""

    def get_decisions_ranked(self, context: AHContext) -> list[str] | None:
        return None


class LearnedHeuristicRegression(LearnedHeuristic):
    def get_feedback(self, context: AHContext, choice: Choice) -> float:
        return 1.0

    def get_decision(self, context: AHContext, choices: list[Choice]) -> Choice | None:
        choice2feedback = {}
        for choice in choices:
            predicted_feedback = self.get_feedback(context, choice)
            choice2feedback[choice] = predicted_feedback
        sorted_choices_feedback = sorted(
            choice2feedback.items(), key=operator.itemgetter(1)
        )
        highest_feedback = sorted_choices_feedback[-1][1]
        second_highest_feedback = sorted_choices_feedback[-2][1]
        if highest_feedback / second_highest_feedback > self.get_confidence_threshold():
            return sorted_choices_feedback[-1][0]
        # We are not sure which choice is the best one
        return None


class LearnedHeuristicDecision(LearnedHeuristic):
    def get_choice(self, idx: int) -> str | None:
        return None

    def get_decision(self, context: AHContext, choices: list[Choice]) -> Choice | None:
        best_choices = self.get_best_choices(context)
        if not best_choices:
            return None
        (best_choice_proba, best_choice_idx) = best_choices[0]
        if best_choice_proba <= self.get_confidence_threshold():
            return None
        return self.get_choice(best_choice_idx)

    def get_decisions_ranked(self, context: AHContext) -> list[str] | None:
        feedback_idx_list = self.get_best_choices(context)
        if feedback_idx_list is None:
            return None
        choices = [
            self.get_choice(feedback_idx[1]) for feedback_idx in feedback_idx_list
        ]
        choices = [choice for choice in choices if choice is not None]
        return choices

    def get_best_choices(self, context: AHContext) -> list[tuple[float, int]] | None:
        return []
