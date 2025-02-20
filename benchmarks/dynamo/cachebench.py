import argparse
import dataclasses
import json
import logging
import os
import subprocess
import sys
import tempfile

from torch._inductor.utils import fresh_inductor_cache


logger: logging.Logger = logging.getLogger(__name__)

TIMEOUT: int = 2000

MODELS: list[str] = ["nanogpt", "BERT_pytorch", "resnet50"]


@dataclasses.dataclass
class RunResult:
    model: str
    mode: str
    dynamic: bool
    cold_compile_s: float
    warm_compile_s: float
    speedup: float


def get_compile_time(file: tempfile._TemporaryFileWrapper) -> float:
    lines = file.readlines()
    # Decode from byte string, remove new lines, parse csv
    lines = [line.decode("utf-8").strip().split(",") for line in lines]
    compilation_time_idx = lines[0].index("compilation_latency")
    compilation_time = lines[1][compilation_time_idx]
    return float(compilation_time)


def _run_torchbench_from_args(model: str, args: list[str]) -> tuple[float, float]:
    with fresh_inductor_cache():
        env = os.environ.copy()
        with tempfile.NamedTemporaryFile(suffix=".csv") as file:
            args.append("--output=" + file.name)
            logger.info(f"Performing cold-start run for {model}")  # noqa: G004
            subprocess.check_call(args, timeout=TIMEOUT, env=env)
            cold_compile_time = get_compile_time(file)

        args.pop()
        with tempfile.NamedTemporaryFile(suffix=".csv") as file:
            args.append("--output=" + file.name)
            logger.info(f"Performing warm-start run for {model}")  # noqa: G004
            subprocess.check_call(args, timeout=TIMEOUT, env=env)
            warm_compile_time = get_compile_time(file)

        return cold_compile_time, warm_compile_time


def _run_torchbench_model(results: list[RunResult], model: str) -> None:
    cur_file = os.path.abspath(__file__)
    torchbench_file = os.path.join(os.path.dirname(cur_file), "torchbench.py")
    assert os.path.exists(
        torchbench_file
    ), f"Torchbench does not exist at {torchbench_file}"

    base_args = [
        sys.executable,
        torchbench_file,
        f"--only={model}",
        "--repeat=1",
        "--performance",
        "--backend=inductor",
    ]
    for mode, mode_args in [
        ("inference", ["--inference", "--bfloat16"]),
        ("training", ["--training", "--amp"]),
    ]:
        for dynamic, dynamic_args in [
            (False, []),
            (True, ["--dynamic-shapes", "--dynamic-batch-only"]),
        ]:
            args = list(base_args)
            args.extend(mode_args)
            args.extend(dynamic_args)

            logger.info(f"Command: {args}")  # noqa: G004
            try:
                cold_compile_t, warm_compile_t = _run_torchbench_from_args(model, args)
                results.append(
                    RunResult(
                        "model",
                        mode,
                        dynamic,
                        cold_compile_t,
                        warm_compile_t,
                        cold_compile_t / warm_compile_t,
                    )
                )
            except Exception as e:
                print(e)
                return None


def _write_results_to_json(results: list[RunResult], output_filename: str) -> None:
    records = []
    for result in results:
        for metric_name, value in [
            ("cold_compile_time(s)", result.cold_compile_s),
            ("warm_compile_time(s)", result.warm_compile_s),
            ("speedup", result.speedup),
        ]:
            records.append(
                {
                    "benchmark": {
                        "name": "cache_benchmarks",
                        "mode": result.mode,
                        "extra_info": {
                            "is_dynamic": result.dynamic,
                        },
                    },
                    "model": {
                        "name": result.model,
                        "backend": "inductor",
                    },
                    "metric": {
                        "name": metric_name,
                        "type": "OSS model",
                        "benchmark_values": [value],
                    },
                }
            )
    with open(output_filename, "w") as f:
        json.dump(records, f)


def parse_cmd_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a TorchBench ServiceLab benchmark."
    )
    parser.add_argument(
        "-m",
        "--model",
        help="Name of the model to run",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="The output filename (json)",
    )
    args, _ = parser.parse_known_args()
    return args


def main() -> None:
    args = parse_cmd_args()

    results: list[RunResult] = []

    if args.model is not None:
        _run_torchbench_model(results, args.model)
    else:
        for model in MODELS:
            _run_torchbench_model(results, model)
    _write_results_to_json(results, args.output)


if __name__ == "__main__":
    main()
