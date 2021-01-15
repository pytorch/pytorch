"""Unpack GroupedTimerArgs into one or more TimerArgs."""
import itertools as it
import re
from typing import List, Tuple, TYPE_CHECKING

from core.api import Mode, TimerArgs, GroupedTimerArgs
from core.jit import construct_model_invocation, generate_torchscript_file
from core.types import Label, FlatDefinition, FlatIntermediateDefinition
from definitions.setup import SETUP_MAP
from worker.main import CostEstimate, WorkerTimerArgs


if TYPE_CHECKING:
    # See core.api for an explanation.
    from torch.utils.benchmark.utils.timer import Language
else:
    from torch.utils.benchmark import Language


def unpack(definitions: FlatIntermediateDefinition) -> FlatDefinition:
    results: List[Tuple[Label, Mode, TimerArgs]] = []

    for label, args in definitions.items():
        if isinstance(args, TimerArgs):
            mode = (
                Mode.EXPLICIT_PY if args.language == Language.PYTHON else
                Mode.EXPLICIT_CPP)
            results.append((label, mode, args))

        else:
            assert isinstance(args, GroupedTimerArgs)

            # TorchScript model
            name: str = re.sub(r'[^a-z_]', '_', '_'.join(label).lower())
            model_path = generate_torchscript_file(args, name=name)
            ts_mode_map = {Mode.PY: Mode.PY_TS, Mode.CPP: Mode.CPP_TS}

            for stmt, mode, language, global_setup in (
                (args.py_stmt, Mode.PY, Language.PYTHON, None),
                (args.cpp_stmt, Mode.CPP, Language.CPP, args.global_setup),
            ):
                # Eager invocation.
                if stmt is not None:
                    timer_args = TimerArgs(
                        stmt=stmt,
                        setup=SETUP_MAP[args.setup][language],
                        global_setup=global_setup,
                        num_threads=args.num_threads,
                        language=language,
                        cost=args.cost,
                    )
                    results.append((label, mode, timer_args))

                # TorchScript invocation.
                if model_path is not None:
                    signature = args.torchscript_signature
                    assert signature is not None
                    jit_timer_args = construct_model_invocation(
                        model_path=model_path,
                        arguments=signature[0],
                        setup=SETUP_MAP[args.setup][language],
                        global_setup=global_setup,
                        num_threads=args.num_threads,
                        language=language,
                        cost=args.cost,
                    )
                    results.append((label, ts_mode_map[mode], jit_timer_args))

    # Lower from `TimerArgs` to `WorkerTimerArgs`
    second_pass_results: List[Tuple[Label, Mode, WorkerTimerArgs]] = []
    for label, mode, timer_args in results:
        for t_args in timer_args.flatten():
            second_pass_results.append((label, mode, t_args))

    return tuple(second_pass_results)
