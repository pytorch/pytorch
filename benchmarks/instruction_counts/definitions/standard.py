"""Default set of benchmarks."""
import functools

from core.api import Setup, TimerArgs, GroupedTimerArgs
from core.types import FlatIntermediateDefinition
from core.utils import flatten
from worker.main import CostEstimate


BENCHMARKS: FlatIntermediateDefinition = flatten({
    "empty": {
        "no allocation": GroupedTimerArgs(
            r"torch.empty(())",
            r"torch::empty({0});",
            Setup.NONE,
            cost=CostEstimate.LESS_THAN_10_US,
        ),

        "with allocation": GroupedTimerArgs(
            r"torch.empty((1,))",
            r"torch::empty({1});",
            Setup.NONE,
            cost=CostEstimate.LESS_THAN_10_US,
        ),
    },

    ("Pointwise", "Data movement"): {
        "contiguous (trivial)": GroupedTimerArgs(
            r"x.contiguous()",
            r"x.contiguous();",
            Setup.TRIVIAL,
            cost=CostEstimate.LESS_THAN_10_US
        ),

        "contiguous (non-trivial)": GroupedTimerArgs(
            r"x.t().contiguous()",
            r"x.t().contiguous();",
            Setup.TRIVIAL,
            cost=CostEstimate.LESS_THAN_10_US
        ),
    },
})
