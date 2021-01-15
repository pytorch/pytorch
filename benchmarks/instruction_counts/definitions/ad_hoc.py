"""Alternate place to define benchmarks. (e.g. for debugging.)

This file also goes into more detail how to define a benchmark.
"""
from torch.utils.benchmark import Language

from core.api import Setup, TimerArgs, GroupedTimerArgs
from core.types import FlatIntermediateDefinition
from core.utils import flatten
from definitions.standard import BENCHMARKS as STANDARD_BENCHMARKS
from worker.main import CostEstimate


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

    "group definition": GroupedTimerArgs(
        py_stmt="""
            y = x.clone()
            y += 5
        """,
        cpp_stmt="""
            auto y = x.clone();
            y += 5;
        """,

        # Need to add entry to Setup Enum so grouping code knows how to
        # set up in both Python and C++.
        setup=Setup.EXAMPLE_FOR_ADHOC,

        # Optional. This will allow the testing infrastructure to measure
        # TorchScript performance as well.
        signature="f(x) -> y",

        # Optional. Defaults to `CostEstimate.AUTO` Used to determine how many
        # Callgrind iterations to run.
        cost=CostEstimate.AUTO,
    ),

    # Borrow example from the standard set. (e.g. for debugging a known regression.)
    "zero_ (from standard)": STANDARD_BENCHMARKS[("Pointwise", "Data movement", "zero_")],
})
