"""Strict ResNet18 FP32 inference benchmark: eager vs loops_ir vs Inductor.

Genuine local TorchBench model only (no toy substitution):

    Model(test="eval", device="cuda", batch_size=8,
          extra_args=["--precision", "fp32"])

from root /home/lsakka/pytorch10/pytorch/benchmark with standard
IMAGENET1K_V1 weights (cached fixture). Same module weights and same
inputs are shared across all three modes under ``torch.inference_mode``.

Determinism (before first GPU work): ``CUBLAS_WORKSPACE_CONFIG=:4096:8``
if unset, ``torch.use_deterministic_algorithms(True)``,
``cudnn.deterministic=True``, ``cudnn.benchmark=False``,
``matmul/cudnn.allow_tf32=False``. ``torch.manual_seed(0)``.

Run command (isolated GPU; visibility is NOT hardcoded in Python)::

    CUDA_VISIBLE_DEVICES=3 <python> \
        /home/lsakka/pytorch10/pytorch/benchmarks/dynamo/loops_ir_benchmark.py \
        [--warmup 20] [--repeats 5] [--output-dir DIR]

Compile: ``fullgraph=True, dynamic=False`` for both compiled modes, normal
launch (same CUDA-graph treatment: none) for all three.

Correctness: eager is authoritative. PyTree structure + dtype + shape
equality first, then contiguous logical values compared as uint8 raw bytes
(not ``torch.equal`` on floats, no tolerance). Inductor bitwise status is
reported separately. If new-backend bits mismatch, the report records
failure -- never relaxes the assert or claims success. Partial/error
reports are allowed but failed runs are never timed as valid.

Timing: vetted fixed-input pure inference via
``torch._inductor.runtime.benchmarking.benchmarker.benchmark_gpu`` with the
REQUIRED explicit ``is_vetted_benchmarking=True`` flag under deterministic
mode (otherwise it raises via ``may_ban_benchmarking``). Compile/warmup are
outside measurements; CLI ``--warmup``/``--repeats`` default to 20/5.
Median and min over repeats are reported.

Artifacts (fresh output dir, default ``tempfile.mkdtemp``): LoopsIR pass
dumps, KernelIR dump, TiledIR dump, each Triton source, available PTX, plus
a structured JSON report (``report.json``) with model/config/versions/
device/batch/seed, coverage counts (extern/generated/fused), bytewise
results (eager-repeat, new-vs-eager, Inductor-vs-eager, optional
new-vs-Inductor), shapes/dtypes, and repeated latencies.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import statistics
import sys
import tempfile
from pathlib import Path
from typing import Any

BENCHMARK_ROOT = "/home/lsakka/pytorch10/pytorch/benchmark"
HARNESS_PATH = "/home/lsakka/pytorch10/pytorch/benchmarks/dynamo/loops_ir_benchmark.py"
BACKEND_PATH = "/home/lsakka/pytorch10/pytorch/torch/_inductor/loops_ir_backend.py"
LOWERING_PATH = "/home/lsakka/pytorch10/pytorch/torch/_inductor/loops_ir.py"
WEIGHTS_FIXTURE = "/home/lsakka/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth"

MODEL_NAME = "resnet18"
BATCH_SIZE = 8
SEED = 0
TEST_MODE = "eval"
DEVICE = "cuda"
EXTRA_ARGS = ["--precision", "fp32"]


def _configure_determinism() -> None:
    import torch

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(SEED)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception as error:
        raise RuntimeError(f"deterministic algorithms unavailable: {error}") from error
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
    except Exception:
        pass
    try:
        torch.backends.cudnn.allow_tf32 = False
    except Exception:
        pass


def _load_model() -> tuple[Any, tuple[Any, ...]]:
    """Load the genuine TorchBench ResNet18 eval module + example inputs."""
    if BENCHMARK_ROOT not in sys.path:
        sys.path.insert(0, BENCHMARK_ROOT)
    try:
        from torchbenchmark.models.resnet18 import Model
    except Exception as error:
        raise RuntimeError(
            f"cannot import genuine TorchBench resnet18 Model "
            f"from {BENCHMARK_ROOT}: {error}"
        ) from error
    try:
        benchmark = Model(
            test=TEST_MODE,
            device=DEVICE,
            batch_size=BATCH_SIZE,
            extra_args=list(EXTRA_ARGS),
        )
    except Exception as error:
        raise RuntimeError(f"TorchBench Model construction failed: {error}") from error
    model, example_inputs = benchmark.get_module()
    model.eval()
    if tuple(model.parameters()).__len__() == 0:
        raise RuntimeError("loaded model has no parameters (not the real ResNet18?)")
    n_params = sum(p.numel() for p in model.parameters())
    if n_params < 10_000_000:
        raise RuntimeError(
            f"loaded model has only {n_params} params; refusing toy substitution"
        )
    inputs = tuple(t.to(DEVICE) if hasattr(t, "to") else t for t in example_inputs)
    return model, inputs


def _raw_bytes(tensor: Any) -> bytes:
    import torch

    return (
        tensor.detach()
        .contiguous()
        .reshape(-1)
        .view(torch.uint8)
        .cpu()
        .numpy()
        .tobytes()
    )


@dataclasses.dataclass
class _CompareResult:
    structures_equal: bool
    dtypes_equal: bool
    shapes_equal: bool
    bytewise_equal: bool
    max_abs_diff: float | None
    detail: str = ""


def _compare_outputs(actual: Any, expected: Any, label: str) -> _CompareResult:
    """PyTree structure+dtype+shape equality, then uint8 byte comparison."""
    import torch
    from torch.utils._pytree import tree_flatten

    try:
        flat_actual, spec_actual = tree_flatten(actual)
        flat_expected, spec_expected = tree_flatten(expected)
    except Exception as error:
        return _CompareResult(
            False, False, False, False, None, f"{label}: flatten: {error}"
        )
    if spec_actual != spec_expected:
        return _CompareResult(
            False, False, False, False, None, f"{label}: pytree structure mismatch"
        )
    if len(flat_actual) != len(flat_expected):
        return _CompareResult(
            True, False, False, False, None, f"{label}: leaf count mismatch"
        )
    for leaf in (*flat_actual, *flat_expected):
        if not isinstance(leaf, torch.Tensor):
            return _CompareResult(
                True, False, False, False, None, f"{label}: non-tensor leaf"
            )
    dtypes_equal = all(a.dtype == b.dtype for a, b in zip(flat_actual, flat_expected))
    shapes_equal = all(
        tuple(a.shape) == tuple(b.shape) and tuple(a.stride()) == tuple(b.stride())
        for a, b in zip(flat_actual, flat_expected)
    )
    if not (dtypes_equal and shapes_equal):
        return _CompareResult(
            True,
            dtypes_equal,
            shapes_equal,
            False,
            None,
            f"{label}: dtype/shape mismatch",
        )
    max_abs: float = 0.0
    all_bytes_equal = True
    for a, b in zip(flat_actual, flat_expected):
        if _raw_bytes(a) != _raw_bytes(b):
            all_bytes_equal = False
        try:
            diff = (
                (a.detach().float().cpu() - b.detach().float().cpu()).abs().max().item()
            )
            max_abs = max(max_abs, float(diff))
        except Exception:
            pass
    detail = f"{label}: {'BYTEWISE-EQUAL' if all_bytes_equal else 'BYTE-MISMATCH'}"
    return _CompareResult(True, True, True, all_bytes_equal, max_abs, detail)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Strict ResNet18 loops_ir benchmark")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default="")
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.repeats < 1:
        parser.error("warmup must be nonnegative and repeats must be positive")

    _configure_determinism()

    import torch

    if not torch.cuda.is_available():
        print("CUDA is required", file=sys.stderr)
        return 2

    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(tempfile.mkdtemp(prefix="loops_ir_bench_"))
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = out_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    print(f"output_dir={out_dir}", flush=True)

    report: dict[str, Any] = {
        "model": MODEL_NAME,
        "test": TEST_MODE,
        "device": DEVICE,
        "batch_size": BATCH_SIZE,
        "extra_args": EXTRA_ARGS,
        "seed": SEED,
        "harness": HARNESS_PATH,
        "backend": BACKEND_PATH,
        "cpp_ir_integration": {
            "backend": "inductor",
            "options": {"experimental_cpp_ir": True, "triton.cudagraphs": False},
            "runtime": "CompiledFxGraph",
            "fresh_compilation_for_artifacts": True,
        },
        "lowering": LOWERING_PATH,
        "benchmark_root": BENCHMARK_ROOT,
        "weights_fixture": WEIGHTS_FIXTURE,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "output_dir": str(out_dir),
        "errors": [],
    }

    model, inputs = _load_model()
    model = model.to(DEVICE)
    _configure_determinism()
    if any(
        p.is_floating_point() and p.dtype != torch.float32 for p in model.parameters()
    ):
        raise RuntimeError("TorchBench model parameters must be FP32")
    report["numerical_settings"] = {
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "compiler_cuda_graphs": False,
    }
    report["num_parameters"] = sum(p.numel() for p in model.parameters())
    report["input_shapes"] = [tuple(t.shape) for t in inputs]
    report["input_dtypes"] = [str(t.dtype) for t in inputs]

    # -- eager reference (authoritative) -----------------------------------
    with torch.inference_mode():
        eager_out = model(*inputs)
        eager_repeat = model(*inputs)
    eager_self = _compare_outputs(eager_repeat, eager_out, "eager-repeat")
    report["eager_repeat_bytewise"] = eager_self.bytewise_equal
    if not eager_self.bytewise_equal:
        report["errors"].append(
            "eager is not repeat-bitwise; environment nondeterministic"
        )

    # -- optional C++ IR compile in Inductor (outside measurements) ---------
    from torch._inductor.loops_ir_backend import LoopsIrArtifacts, loops_ir_backend

    collector = LoopsIrArtifacts()
    new_error: str | None = None
    new_fn = None
    try:
        new_fn = torch.compile(
            model,
            backend=lambda gm, ex: loops_ir_backend(
                gm,
                ex,
                dumps=True,
                source=True,
                ir=True,
                statistics=True,
                collector=collector,
            ),
            fullgraph=True,
            dynamic=False,
        )
        # Artifact collection is a diagnostic side channel, not a cache-key
        # input. Compile freshly so a cache hit cannot empty its coverage count.
        with (
            torch._inductor.config.patch(
                {"fx_graph_cache": False, "fx_graph_remote_cache": False}
            ),
            torch.inference_mode(),
        ):
            new_out = new_fn(*inputs)
    except Exception as error:
        new_error = f"{type(error).__name__}: {error}"
        report["errors"].append(f"loops_ir compile/run failed: {new_error}")
        new_out = None

    # Generated-kernel coverage gate: fail loudly when nothing generated.
    report["coverage"] = {
        "num_generated_kernels": collector.num_generated_kernels,
        "kernels_before_fusion": collector.kernels_before_fusion,
        "kernels_removed_by_fusion": collector.kernels_before_fusion
        - collector.num_generated_kernels,
        "num_extern_calls": collector.num_extern_calls,
        "function_names": list(collector.function_names),
    }
    if new_error is None and collector.num_generated_kernels <= 0:
        report["errors"].append(
            "GENERATEDKERNEL count is 0: no generated kernels executed; failing loudly"
        )
        new_error = "no generated kernels (coverage gate)"

    # -- Inductor compile (outside measurements) -----------------------------
    inductor_error: str | None = None
    inductor_fn = None
    inductor_out = None
    try:
        inductor_fn = torch.compile(
            model,
            backend="inductor",
            fullgraph=True,
            dynamic=False,
            options={"triton.cudagraphs": False},
        )
        with torch.inference_mode():
            inductor_out = inductor_fn(*inputs)
    except Exception as error:
        inductor_error = f"{type(error).__name__}: {error}"
        report["errors"].append(f"inductor compile/run failed: {inductor_error}")

    # -- bytewise comparisons (eager authoritative) ----------------------------
    if new_out is not None:
        new_cmp = _compare_outputs(new_out, eager_out, "new-vs-eager")
        report["new_vs_eager"] = dataclasses.asdict(new_cmp)
    else:
        report["new_vs_eager"] = {"error": new_error}
    if inductor_out is not None:
        ind_cmp = _compare_outputs(inductor_out, eager_out, "inductor-vs-eager")
        report["inductor_vs_eager"] = dataclasses.asdict(ind_cmp)
    else:
        report["inductor_vs_eager"] = {"error": inductor_error}
    if new_out is not None and inductor_out is not None:
        both_cmp = _compare_outputs(new_out, inductor_out, "new-vs-inductor")
        report["new_vs_inductor"] = dataclasses.asdict(both_cmp)

    # -- artifacts -------------------------------------------------------------
    (artifacts_dir / "loops_ir_program.txt").write_text(
        collector.program_dump or "<empty: capture with dumps=True>"
    )
    (artifacts_dir / "loops_ir_optimized.txt").write_text(
        collector.optimized_program_dump or "<empty>"
    )
    (artifacts_dir / "kernel_ir.txt").write_text(collector.kernel_dump or "<empty>")
    (artifacts_dir / "tiled_ir.txt").write_text(collector.tiled_dump or "<empty>")
    for fname, src in collector.triton_sources.items():
        safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in fname)
        (artifacts_dir / f"triton_{safe}.py").write_text(src)
    for fname, ptx in collector.triton_ptx.items():
        safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in fname)
        (artifacts_dir / f"ptx_{safe}.ptx").write_text(ptx)
    report["artifacts"] = sorted(p.name for p in artifacts_dir.iterdir())

    # -- timing (compile/warmup outside; never time failed runs) ----------------
    from torch._inductor.runtime.benchmarking import benchmarker

    def _timed_latencies(fn: Any, label: str) -> dict[str, Any]:
        for _ in range(args.warmup):
            fn(*inputs)
        torch.cuda.synchronize()
        samples: list[float] = []
        for _ in range(args.repeats):
            ms = benchmarker.benchmark_gpu(
                lambda: fn(*inputs), is_vetted_benchmarking=True
            )
            samples.append(float(ms))
        torch.cuda.synchronize()
        return {
            "samples_ms": samples,
            "median_ms": statistics.median(samples),
            "min_ms": min(samples),
            "max_ms": max(samples),
        }

    def _safe_timing(fn: Any, label: str) -> dict[str, Any]:
        try:
            return _timed_latencies(fn, label)
        except Exception as error:
            detail = f"{label} timing failed: {type(error).__name__}: {error}"
            report["errors"].append(detail)
            return {"error": detail}

    with torch.inference_mode():
        report["latency_eager_ms"] = _safe_timing(model, "eager")
        if (
            new_fn is not None
            and new_error is None
            and eager_self.bytewise_equal
            and report["new_vs_eager"].get("bytewise_equal") is True
        ):
            report["latency_loops_ir_ms"] = _safe_timing(new_fn, "loops_ir")
        else:
            report["latency_loops_ir_ms"] = {
                "error": "not timed: failed run is never valid"
            }
        if inductor_fn is not None and inductor_error is None:
            report["latency_inductor_ms"] = _safe_timing(inductor_fn, "inductor")
        else:
            report["latency_inductor_ms"] = {
                "error": "not timed: failed run is never valid"
            }

    # -- verdict -----------------------------------------------------------------
    new_bits = report.get("new_vs_eager", {})
    timings_complete = all(
        "median_ms" in report[key]
        for key in ("latency_eager_ms", "latency_loops_ir_ms", "latency_inductor_ms")
    )
    if new_error is not None:
        report["verdict"] = f"ERROR: {new_error}"
    elif not eager_self.bytewise_equal:
        report["verdict"] = "FAIL: eager repeat is not bitwise deterministic"
    elif not isinstance(new_bits, dict) or new_bits.get("bytewise_equal") is not True:
        report["verdict"] = (
            "FAIL: loops_ir bits differ from eager (reported, not relaxed)"
        )
    elif not timings_complete:
        report["verdict"] = "ERROR: benchmark timings incomplete; see errors"
    else:
        report["verdict"] = "PASS: loops_ir bitwise-equal to eager"

    (out_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"output_dir={out_dir}")
    return 0 if report["verdict"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
