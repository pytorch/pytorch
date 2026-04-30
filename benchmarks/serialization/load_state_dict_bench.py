import argparse
from dataclasses import dataclass

import torch
import torch.utils.benchmark as benchmark


DEFAULT_CHILDREN = (128, 512, 2048)
DEFAULT_PARAMS_PER_CHILD = 2
DEFAULT_PARAMETER_SIZE = 1
DEFAULT_MIN_RUN_TIME = 1.0


class ParameterGroup(torch.nn.Module):
    def __init__(self, params_per_child: int, parameter_size: int) -> None:
        super().__init__()
        for idx in range(params_per_child):
            self.register_parameter(
                f"param{idx}",
                torch.nn.Parameter(torch.zeros(parameter_size)),
            )


class WideModule(torch.nn.Module):
    def __init__(
        self,
        num_children: int,
        params_per_child: int,
        parameter_size: int,
    ) -> None:
        super().__init__()
        for idx in range(num_children):
            self.add_module(
                f"child{idx}",
                ParameterGroup(params_per_child, parameter_size),
            )


@dataclass(frozen=True)
class BenchmarkResult:
    num_children: int
    params_per_child: int
    state_dict_keys: int
    assign: bool
    median_load_ms: float
    iqr_load_ms: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark nn.Module.load_state_dict on wide module hierarchies. "
            "This keeps tensors small to focus on Python state_dict key routing."
        )
    )
    parser.add_argument(
        "--children",
        type=int,
        nargs="+",
        default=list(DEFAULT_CHILDREN),
        help="Number of direct child modules to benchmark.",
    )
    parser.add_argument(
        "--params-per-child",
        type=int,
        default=DEFAULT_PARAMS_PER_CHILD,
        help="Number of parameters registered on each child module.",
    )
    parser.add_argument(
        "--parameter-size",
        type=int,
        default=DEFAULT_PARAMETER_SIZE,
        help="Number of float32 elements in each parameter.",
    )
    parser.add_argument(
        "--min-run-time",
        type=float,
        default=DEFAULT_MIN_RUN_TIME,
        help="Minimum seconds per benchmark case for blocked_autorange.",
    )
    parser.add_argument(
        "--assign",
        action="store_true",
        help="Benchmark load_state_dict(..., assign=True).",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if any(num_children <= 0 for num_children in args.children):
        raise ValueError("--children values must be positive")
    if args.params_per_child <= 0:
        raise ValueError("--params-per-child must be positive")
    if args.parameter_size <= 0:
        raise ValueError("--parameter-size must be positive")
    if args.min_run_time <= 0:
        raise ValueError("--min-run-time must be positive")


def _make_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.clone() for key, value in module.state_dict().items()}


def _run_case(
    *,
    num_children: int,
    params_per_child: int,
    parameter_size: int,
    assign: bool,
    min_run_time: float,
) -> BenchmarkResult:
    module = WideModule(num_children, params_per_child, parameter_size)
    state_dict = _make_state_dict(module)

    module.load_state_dict(state_dict, assign=assign)
    timer = benchmark.Timer(
        stmt="module.load_state_dict(state_dict, assign=assign)",
        globals={
            "module": module,
            "state_dict": state_dict,
            "assign": assign,
        },
        label="nn.Module.load_state_dict",
        sub_label="wide module hierarchy",
        description=f"children={num_children}, keys={len(state_dict)}",
    )
    measurement = timer.blocked_autorange(min_run_time=min_run_time)
    return BenchmarkResult(
        num_children=num_children,
        params_per_child=params_per_child,
        state_dict_keys=len(state_dict),
        assign=assign,
        median_load_ms=measurement.median * 1000.0,
        iqr_load_ms=measurement.iqr * 1000.0,
    )


def _format_table(results: list[BenchmarkResult]) -> str:
    headers = (
        "children",
        "params_per_child",
        "keys",
        "assign",
        "median_load_ms",
        "iqr_load_ms",
    )
    rows = [
        (
            str(result.num_children),
            str(result.params_per_child),
            str(result.state_dict_keys),
            str(result.assign),
            f"{result.median_load_ms:.3f}",
            f"{result.iqr_load_ms:.3f}",
        )
        for result in results
    ]
    widths = [
        max(len(row[idx]) for row in (headers, *rows))
        for idx in range(len(headers))
    ]

    def format_row(row: tuple[str, ...]) -> str:
        return "  ".join(value.rjust(width) for value, width in zip(row, widths))

    separator = format_row(tuple("-" * width for width in widths))
    return "\n".join(
        [format_row(headers), separator] + [format_row(row) for row in rows]
    )


def main() -> None:
    args = _parse_args()
    _validate_args(args)
    results = [
        _run_case(
            num_children=num_children,
            params_per_child=args.params_per_child,
            parameter_size=args.parameter_size,
            assign=args.assign,
            min_run_time=args.min_run_time,
        )
        for num_children in args.children
    ]
    print(_format_table(results))


if __name__ == "__main__":
    main()
