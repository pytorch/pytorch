import argparse
import gc
import statistics
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import torch


DEFAULT_NUM_PARAMS = (
    1_000_000,
    10_000_000,
    100_000_000,
    1_000_000_000,
    2_000_000_000,
    3_000_000_000,
    4_000_000_000,
)
NUM_LAYERS = 5
DEFAULT_REPEATS = 5


class LinearModel(torch.nn.Module):
    def __init__(self, hidden_size: int, *, dtype: torch.dtype) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(
                    hidden_size,
                    hidden_size,
                    bias=True,
                    dtype=dtype,
                )
                for _ in range(NUM_LAYERS)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


@dataclass
class BenchmarkResult:
    num_params: int
    median_save_ms: float | None
    status: str
    error_detail: str | None = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark torch.export.save for a five-layer LinearModel across "
            "a range of parameter counts."
        )
    )
    parser.add_argument(
        "--num-params",
        type=int,
        nargs="+",
        default=list(DEFAULT_NUM_PARAMS),
        help=(
            "Target total parameter counts to benchmark. The script derives the "
            "nearest hidden size for each target."
        ),
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help="Number of torch.export.save timings to collect per parameter-count case.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help=(
            "Batch size for the example input passed to torch.export.export. "
            "The default of 0 keeps the benchmark focused on serialization cost."
        ),
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
        help="Parameter and example-input dtype.",
    )
    return parser.parse_args()


def _dtype_from_name(dtype_name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype_name]


def _parameter_count(hidden_size: int) -> int:
    return NUM_LAYERS * (hidden_size * hidden_size + hidden_size)


def _hidden_size_from_num_params(num_params: int) -> int:
    if num_params <= 0:
        raise ValueError("--num-params values must be positive")

    root = (-1.0 + (1.0 + 4.0 * num_params / NUM_LAYERS) ** 0.5) / 2.0
    base_hidden_size = max(1, int(root))
    candidates = [
        candidate
        for candidate in (
            base_hidden_size - 1,
            base_hidden_size,
            base_hidden_size + 1,
            base_hidden_size + 2,
        )
        if candidate >= 1
    ]
    return min(
        candidates,
        key=lambda hidden_size: abs(_parameter_count(hidden_size) - num_params),
    )


def _format_num_params(num_params: int) -> str:
    for threshold, suffix in (
        (1_000_000_000, "B"),
        (1_000_000, "M"),
        (1_000, "K"),
    ):
        if num_params >= threshold:
            value = num_params / threshold
            return f"{value:.1f}".rstrip("0").rstrip(".") + suffix
    return str(num_params)


def _measure_save_times_ms(
    exported_program: torch.export.ExportedProgram,
    repeats: int,
) -> list[float]:
    times_ms: list[float] = []
    with tempfile.TemporaryDirectory(prefix="export_save_bench_") as temp_dir:
        temp_path = Path(temp_dir)
        for iteration in range(repeats):
            save_path = temp_path / f"exported_program_{iteration}.pt2"
            start_time = time.perf_counter()
            torch.export.save(exported_program, save_path)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            times_ms.append(elapsed_ms)
            save_path.unlink(missing_ok=True)
    return times_ms


def _status_for_exception(exc: BaseException) -> tuple[str, str]:
    message = (
        str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
    )
    lowered = message.lower()
    if (
        isinstance(exc, MemoryError)
        or "out of memory" in lowered
        or "can't allocate memory" in lowered
    ):
        return "oom", message
    return "error", message


def _run_case(
    target_num_params: int,
    *,
    repeats: int,
    batch_size: int,
    dtype: torch.dtype,
) -> BenchmarkResult:
    exported_program = None
    hidden_size = _hidden_size_from_num_params(target_num_params)
    num_params = _parameter_count(hidden_size)
    try:
        with torch.no_grad():
            model = LinearModel(hidden_size, dtype=dtype).eval()
            model.requires_grad_(False)
            example_input = torch.zeros(batch_size, hidden_size, dtype=dtype)
            exported_program = torch.export.export(
                model,
                (example_input,),
                strict=True,
            )

        save_times_ms = _measure_save_times_ms(exported_program, repeats)
        return BenchmarkResult(
            num_params=num_params,
            median_save_ms=statistics.median(save_times_ms),
            status="ok",
        )
    except (MemoryError, RuntimeError) as exc:
        status, detail = _status_for_exception(exc)
        return BenchmarkResult(
            num_params=num_params,
            median_save_ms=None,
            status=status,
            error_detail=detail,
        )
    finally:
        del exported_program
        del example_input
        del model
        gc.collect()


def _format_table(results: list[BenchmarkResult]) -> str:
    headers = ("num_params", "median_save_ms", "status")
    rows = [
        (
            _format_num_params(result.num_params),
            (
                f"{result.median_save_ms:.3f}"
                if result.median_save_ms is not None
                else "n/a"
            ),
            result.status,
        )
        for result in results
    ]
    widths = [
        max(len(header), *(len(row[column]) for row in rows))
        for column, header in enumerate(headers)
    ]

    def _format_row(row: tuple[str, ...]) -> str:
        return " | ".join(
            [
                row[0].rjust(widths[0]),
                row[1].rjust(widths[1]),
                row[2].ljust(widths[2]),
            ]
        )

    lines = [
        _format_row(headers),
        "-+-".join("-" * width for width in widths),
    ]
    lines.extend(_format_row(row) for row in rows)
    return "\n".join(lines)


def main() -> None:
    args = _parse_args()
    dtype = _dtype_from_name(args.dtype)

    print("Benchmarking torch.export.save for a 5-layer LinearModel")
    print(f"dtype={args.dtype}, batch_size={args.batch_size}, repeats={args.repeats}")

    results = [
        _run_case(
            num_params,
            repeats=args.repeats,
            batch_size=args.batch_size,
            dtype=dtype,
        )
        for num_params in args.num_params
    ]

    print()
    print(
        "Note: This benchmark is highly sensitive to disk performance and OS "
        "behavior. Results for smaller parameter counts can be noisy or flaky."
    )
    print()
    print(_format_table(results))

    failures = [result for result in results if result.error_detail is not None]
    if failures:
        print()
        print("failed cases:")
        for result in failures:
            print(
                f"- num_params={_format_num_params(result.num_params)}: "
                f"{result.error_detail}"
            )


if __name__ == "__main__":
    main()
