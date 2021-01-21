"""Default set of benchmarks."""
from core.api_impl import GroupedStmts
from core.types import FlatIntermediateDefinition
from core.utils import flatten
from definitions.setup import Setup


BENCHMARKS: FlatIntermediateDefinition = flatten({
    "empty": {
        "no allocation": GroupedStmts(
            r"torch.empty(())",
            r"torch::empty({0});",
        ),

        "with allocation": GroupedStmts(
            r"torch.empty((1,))",
            r"torch::empty({1});",
        ),
    },

    ("Pointwise", "Data movement"): {
        "contiguous (trivial)": GroupedStmts(
            r"x.contiguous()",
            r"x.contiguous();",
            Setup.TRIVIAL_2D.value,
        ),

        "contiguous (non-trivial)": GroupedStmts(
            r"x.t().contiguous()",
            r"x.t().contiguous();",
            Setup.TRIVIAL_2D.value,
        ),
    },
})
