from typing import Any, List, Optional, Tuple

from torch._inductor.autoheuristic_utils import Choice, ContextDictT


class LearnedHeuristic:
    def __init__(self) -> None:
        pass

    def check_precondition(
        self,
        name: str,
        context_dict: ContextDictT,
        shared_memory: Any,
        device_capa: Tuple[int, int],
    ) -> bool:
        return True

    def get_decision(
        self, context_dict: ContextDictT, choices: List[Choice]
    ) -> Optional[Choice]:
        choice2feedback = {}
        for choice in choices:
            context_dict["choice"] = choice
            predicted_feedback = self.get_feedback(context_dict, choice)
            choice2feedback[choice] = predicted_feedback
        sorted_choices_feedback = sorted(choice2feedback.items(), key=lambda t: t[1])
        highest_feedback = sorted_choices_feedback[-1][1]
        second_highest_feedback = sorted_choices_feedback[-2][1]
        if highest_feedback / second_highest_feedback > self.get_speedup_threshold():
            return sorted_choices_feedback[-1][0]
        # We are not sure which choice is the best one
        return None

    def get_feedback(self, context_dict: ContextDictT, choice: Choice) -> float:
        return 1.0

    def get_speedup_threshold(self) -> float:
        return 1.0

    def get_name(self) -> str:
        return ""
