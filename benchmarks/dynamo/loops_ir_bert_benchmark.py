"""Matched genuine TorchBench BERT inference: eager, stock and C++ IR modes.

The workload is BERT_pytorch's unmodified 12-layer FP32 model, with its
standard sequence length 128. Compiler modes never change ambient precision
settings. Compilation is excluded from interleaved GPU latency measurements.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
from torch._inductor import config
from torch._inductor.runtime.benchmarking import benchmarker
from torch.utils._pytree import tree_map

from benchmarks.dynamo.loops_ir_benchmark import _compare_outputs


BENCHMARK_ROOT = Path(__file__).resolve().parents[2] / "benchmark"


def numerical_settings() -> dict[str, Any]:
    return {
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "emulate_precision_casts": config.emulate_precision_casts,
        "disable_ftz": config.eager_numerics.disable_ftz,
        "use_fast_math": config.use_fast_math,
    }


def load_model(batch_size: int) -> tuple[Any, tuple[torch.Tensor, ...]]:
    if str(BENCHMARK_ROOT) not in sys.path:
        sys.path.insert(0, str(BENCHMARK_ROOT))
    from torchbenchmark.models.BERT_pytorch import Model

    benchmark = Model(
        test="eval",
        device="cuda",
        batch_size=batch_size,
        extra_args=["--precision", "fp32"],
    )
    model, inputs = benchmark.get_module()
    model.eval()
    inputs = tuple(inputs)
    if sum(parameter.numel() for parameter in model.parameters()) < 100_000_000:
        raise RuntimeError("refusing a reduced or substitute BERT workload")
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise RuntimeError("the C++ IR benchmark requires FP32 parameters")
    if any(value.device.type != "cuda" for value in inputs):
        raise RuntimeError("all BERT inputs must be CUDA tensors")
    if any(tuple(value.shape) != (batch_size, 128) for value in inputs):
        raise RuntimeError("expected the genuine BERT_pytorch sequence length 128")
    return model, inputs


def save_artifacts(directory: Path, artifacts: Any) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    dumps = {
        "loops_ir.txt": artifacts.program_dump,
        "optimized_loops_ir.txt": artifacts.optimized_program_dump,
        "kernel_ir.txt": artifacts.kernel_dump,
        "tiled_ir.txt": artifacts.tiled_dump,
    }
    for name, text in dumps.items():
        (directory / name).write_text(text)
    for name, text in artifacts.triton_sources.items():
        (directory / f"{name}.py").write_text(text)
    for name, text in artifacts.triton_ptx.items():
        (directory / f"{name}.ptx").write_text(text)
    return {
        "num_generated_kernels": artifacts.num_generated_kernels,
        "kernels_before_fusion": artifacts.kernels_before_fusion,
        "kernels_removed_by_fusion": (
            artifacts.kernels_before_fusion - artifacts.num_generated_kernels
        ),
        "num_extern_calls": artifacts.num_extern_calls,
        "function_names": list(artifacts.function_names),
        "directory": str(directory),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument(
        "--cpp-numerics", choices=("both", "strict_eager", "inductor"), default="both"
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.warmup < 0 or args.repeats < 1:
        parser.error("batch size/repeats must be positive; warmup nonnegative")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    output_dir = args.output_dir or Path(tempfile.mkdtemp(prefix="loops_ir_bert_"))
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    initial_settings = numerical_settings()
    model, inputs = load_model(args.batch_size)
    settings = numerical_settings()
    report: dict[str, Any] = {
        "model": "TorchBench BERT_pytorch",
        "benchmark_root": str(BENCHMARK_ROOT),
        "batch_size": args.batch_size,
        "sequence_length": 128,
        "layers": 12,
        "hidden_size": 768,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "dtype": "torch.float32",
        "device": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "seed": 0,
        "settings_before_model_construction": initial_settings,
        "effective_settings": settings,
        "fullgraph": True,
        "dynamic": False,
        "compiler_cudagraphs": False,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "accuracy_tolerance": {"rtol": 1e-4, "atol": 1e-5},
        "results": {},
        "errors": [],
        "remaining_optimizations": [
            "generated corrected-std/Welford and softmax reductions",
            "reduction/pointwise and broadcast-producer fusion",
            "GEMM layout and algorithm parity",
            "graph-level lifetime/alias-aware buffer reuse",
            "per-kernel tile selection/autotuning",
            "validated CUDA-graph support",
        ],
    }
    from torch._inductor.loops_ir_backend import (
        _Collector,
        _current_collector,
        LoopsIrArtifacts,
    )

    modes = (
        ("strict_eager", "inductor")
        if args.cpp_numerics == "both"
        else (args.cpp_numerics,)
    )
    functions: dict[str, Any] = {"eager": model}
    with torch.inference_mode():
        reference = tree_map(
            lambda value: value.clone() if isinstance(value, torch.Tensor) else value,
            model(*inputs),
        )
        self_comparison = _compare_outputs(model(*inputs), reference, "eager-repeat")
        report["eager_repeat"] = dataclasses.asdict(self_comparison)
        if not self_comparison.bytewise_equal:
            report["errors"].append("eager reference is not repeat-bitwise stable")
        report["results"]["eager"] = {"accuracy_pass": True}
        cases = [("stock_inductor", None)] + [(f"cpp_{mode}", mode) for mode in modes]
        for label, mode in cases:
            artifacts = LoopsIrArtifacts()
            options: dict[str, Any] = {
                "experimental_cpp_ir": mode is not None,
                "triton.cudagraphs": False,
            }
            if mode is not None:
                options["experimental_cpp_ir_numerics"] = mode
            print(f"compiling {label}", flush=True)
            token = None
            result: dict[str, Any] = {"options": options}
            report["results"][label] = result
            try:
                compiled = torch.compile(
                    model,
                    backend="inductor",
                    fullgraph=True,
                    dynamic=False,
                    options=options,
                )
                if mode is not None:
                    token = _current_collector.set(
                        _Collector(artifacts, dumps=True, source=True, ir=True)
                    )
                start = time.perf_counter()
                # Collect real IR even when a previous benchmark populated caches.
                with config.patch(
                    {"fx_graph_cache": False, "fx_graph_remote_cache": False}
                ):
                    actual = compiled(*inputs)
                torch.cuda.synchronize()
                compile_seconds = time.perf_counter() - start
                comparison = _compare_outputs(actual, reference, label)
                result.update(
                    {
                        "compile_and_first_run_seconds": compile_seconds,
                        "versus_eager": dataclasses.asdict(comparison),
                    }
                )
                if mode is not None:
                    result["coverage"] = save_artifacts(output_dir / label, artifacts)
                    if artifacts.num_generated_kernels < 1:
                        raise RuntimeError("no generated C++ IR kernels executed")
                if mode == "strict_eager":
                    if (
                        not self_comparison.bytewise_equal
                        or not comparison.bytewise_equal
                    ):
                        raise RuntimeError(
                            "strict C++ IR output differs from eager bits"
                        )
                else:
                    torch.testing.assert_close(
                        actual, reference, rtol=1e-4, atol=1e-5, check_stride=True
                    )
                del actual
                if numerical_settings() != settings:
                    raise RuntimeError("compiler changed ambient numerical settings")
                result["accuracy_pass"] = True
                report["results"][label] = result
                functions[label] = compiled
                print(f"accuracy passed: {label}", flush=True)
            except Exception as error:
                result.update(
                    {
                        "accuracy_pass": False,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
                if mode is not None:
                    report["results"][label]["coverage"] = save_artifacts(
                        output_dir / label, artifacts
                    )
                report["errors"].append(f"{label}: {error}")
                print(f"failed: {label}: {error}", flush=True)
            finally:
                if token is not None:
                    _current_collector.reset(token)
        for function in functions.values():
            for _ in range(args.warmup):
                function(*inputs)
        torch.cuda.synchronize()
        samples: dict[str, list[float]] = {label: [] for label in functions}
        labels = list(functions)
        for repeat in range(args.repeats):
            order = labels[repeat % len(labels) :] + labels[: repeat % len(labels)]
            for label in order:
                function = functions[label]
                samples[label].append(
                    float(
                        benchmarker.benchmark_gpu(
                            lambda: function(*inputs), is_vetted_benchmarking=True
                        )
                    )
                )
        eager_ms = statistics.median(samples["eager"])
        for label, values in samples.items():
            report["results"][label]["latency"] = {
                "samples_ms": values,
                "median_ms": statistics.median(values),
                "min_ms": min(values),
                "max_ms": max(values),
                "speedup_over_eager": eager_ms / statistics.median(values),
            }
    report["verdict"] = "PASS" if not report["errors"] else "FAIL"
    report["output_dir"] = str(output_dir)
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
