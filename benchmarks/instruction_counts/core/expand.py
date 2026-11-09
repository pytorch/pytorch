"""Logic for converting human-readable benchmarks into executable form."""

# mypy: ignore-errors

import itertools as it
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    # See the note in api.py for why this is necessary.
    from torch.utils.benchmark.utils.timer import Language
else:
    from torch.utils.benchmark import Language

from core.api import AutogradMode, AutoLabels, GroupedBenchmark, RuntimeMode, TimerArgs
from core.types import FlatDefinition, FlatIntermediateDefinition, Label


_ALL_MODES = tuple(
    it.product(
        RuntimeMode,
        AutogradMode,
        Language,
    )
)


def _get_stmt(
    benchmark: GroupedBenchmark,
    autograd: AutogradMode,
    language: Language,
) -> str | None:
    """Specialize a GroupedBenchmark for a particular configuration."""
    stmt = (
        benchmark.py_fwd_stmt if language == Language.PYTHON else benchmark.cpp_fwd_stmt
    )

    if autograd == AutogradMode.FORWARD_BACKWARD and stmt is not None:
        if benchmark.signature_output is None:
            raise AssertionError(
                "benchmark.signature_output must not be None for FORWARD_BACKWARD mode"
            )
        backward = (
            f"{benchmark.signature_output}"
            f".backward(){';' if language == Language.CPP else ''}"
        )
        stmt = f"{stmt}\n{backward}"
    return stmt


def _get_setup(benchmark: GroupedBenchmark, language: Language) -> str:
    """Collect setup and model definitions for a particular language."""
    if language == Language.PYTHON:
        setup = benchmark.setup.py_setup
        model_setup = benchmark.py_model_setup
    else:
        if language != Language.CPP:
            raise AssertionError(f"Expected Language.CPP, but got {language}")
        setup = benchmark.setup.cpp_setup
        model_setup = benchmark.cpp_model_setup

    return "\n".join([setup, model_setup or ""])


def materialize(benchmarks: FlatIntermediateDefinition) -> FlatDefinition:
    """Split GroupedBenchmarks into TimerArgs and tag the results with AutoLabels."""
    results: list[tuple[Label, AutoLabels, TimerArgs]] = []

    for label, args in benchmarks.items():
        if isinstance(args, TimerArgs):
            # User provided an explicit TimerArgs, so no processing is necessary.
            auto_labels = AutoLabels(
                RuntimeMode.EXPLICIT, AutogradMode.EXPLICIT, args.language
            )
            results.append((label, auto_labels, args))

        else:
            if not isinstance(args, GroupedBenchmark):
                raise AssertionError(f"Expected GroupedBenchmark, but got {type(args)}")

            for (runtime, autograd, language), num_threads in it.product(
                _ALL_MODES, args.num_threads
            ):
                if runtime == RuntimeMode.EXPLICIT or autograd == AutogradMode.EXPLICIT:
                    continue

                if autograd == AutogradMode.FORWARD_BACKWARD and not args.autograd:
                    continue

                stmt = _get_stmt(args, autograd, language)
                if stmt is None:
                    continue

                setup = _get_setup(args, language)

                autolabels = AutoLabels(runtime, autograd, language)
                timer_args = TimerArgs(
                    stmt=stmt,
                    setup=setup,
                    global_setup="",
                    num_threads=num_threads,
                    language=language,
                )

                results.append((label, autolabels, timer_args))

    return tuple(results)
