from __future__ import annotations

from typing import TYPE_CHECKING

from tools.testing.target_determination.heuristics.correlated_with_historical_failures import (
    CorrelatedWithHistoricalFailures,
)
from tools.testing.target_determination.heuristics.edited_by_pr import EditedByPR
from tools.testing.target_determination.heuristics.filepath import Filepath
from tools.testing.target_determination.heuristics.historical_class_failure_correlation import (
    HistoricalClassFailurCorrelation,
)
from tools.testing.target_determination.heuristics.historical_edited_files import (
    HistorialEditedFiles,
)
from tools.testing.target_determination.heuristics.interface import (
    AggregatedHeuristics as AggregatedHeuristics,
    TestPrioritizations as TestPrioritizations,
    TestsToRun as TestsToRun,
)
from tools.testing.target_determination.heuristics.llm import LLM
from tools.testing.target_determination.heuristics.mentioned_in_pr import MentionedInPR
from tools.testing.target_determination.heuristics.previously_failed_in_pr import (
    PreviouslyFailedInPR,
)
from tools.testing.target_determination.heuristics.profiling import Profiling
from tools.testing.target_determination.heuristics.public_bindings import PublicBindings


if TYPE_CHECKING:
    from tools.testing.target_determination.heuristics.interface import (
        HeuristicInterface as HeuristicInterface,
    )


# All currently running heuristics.
# To add a heurstic in trial mode, specify the keyword argument `trial_mode=True`.
HEURISTICS: list[HeuristicInterface] = [
    PreviouslyFailedInPR(),
    EditedByPR(),
    MentionedInPR(),
    HistoricalClassFailurCorrelation(trial_mode=True),
    CorrelatedWithHistoricalFailures(),
    HistorialEditedFiles(),
    Profiling(),
    LLM(),
    Filepath(),
    PublicBindings(),
]
