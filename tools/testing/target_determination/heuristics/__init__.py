from typing import List

from tools.testing.target_determination.heuristics.edited_by_pr import EditedByPR

from tools.testing.target_determination.heuristics.interface import (
    HeuristicInterface as HeuristicInterface,
    TestPrioritizations as TestPrioritizations,
)

# Heuristics
from tools.testing.target_determination.heuristics.previously_failed_in_pr import (
    PreviouslyFailedInPR,
)

HEURISTICS: List[HeuristicInterface] = [
    PreviouslyFailedInPR(),
    EditedByPR(),
]
