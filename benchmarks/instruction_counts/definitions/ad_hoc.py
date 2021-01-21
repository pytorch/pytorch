"""Alternate place to define benchmarks. (e.g. for debugging.)

This file also goes into more detail how to define a benchmark.
"""
from torch.utils.benchmark import Language

from core.api import TimerArgs, GroupedSetup
from core.api_impl import GroupedStmts
from core.types import FlatIntermediateDefinition
from core.utils import flatten
from definitions.standard import BENCHMARKS as STANDARD_BENCHMARKS


ADHOC_BENCHMARKS: FlatIntermediateDefinition = flatten({
    # =========================================================================
    # == Examples =============================================================
    # =========================================================================

    # Skip a lot of infrastructure and just tell the benchmark
    # how to construct a Timer.
    "simple definition": TimerArgs(
        stmt="y = x - 5",
        setup="x = torch.ones((10, 10))",
        language=Language.PYTHON,
    ),

    "group definition": GroupedStmts(
        py_stmt="""
            y = x.clone()
            y += 5
        """,
        cpp_stmt="""
            auto y = x.clone();
            y += 5;
        """,

        setup=GroupedSetup(
            "x = torch.ones((1, 1))",
            "auto x = torch::ones({1, 1});"
        ),

        # Optional. This will allow the testing infrastructure to measure
        # TorchScript performance as well.
        signature="f(x) -> y",
        torchscript=True,
    ),

    # Borrow example from the standard set. (e.g. for debugging a known regression.)
    "zero_ (from standard)": STANDARD_BENCHMARKS[("Pointwise", "Data movement", "zero_")],
})
