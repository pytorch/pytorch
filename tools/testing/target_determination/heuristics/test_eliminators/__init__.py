from typing import List

from tools.testing.target_determination.heuristics import HEURISTICS
from tools.testing.target_determination.heuristics.interface import (
    TargetDeterminatorInterface,
)
from tools.testing.target_determination.heuristics.test_eliminators.remove_unranked_from_heuristics import (
    RemoveUnrankedUsingHeuristics,
)

# TD Methods to test
TD_STRATEGIES: List[TargetDeterminatorInterface] = [
    RemoveUnrankedUsingHeuristics(heuristics=HEURISTICS)
]
