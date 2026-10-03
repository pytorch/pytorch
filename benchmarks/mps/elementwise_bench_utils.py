"""Shared timing and provenance for the two standalone MPS indexing benches."""

import argparse
import contextlib
import hashlib
import json
import os
import platform
import random
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_CONTROLS = ("strided", "ilp", "inner_contiguous", "inner_strided")


@contextlib.contextmanager
def force_flavor(variable, flavor):
    previous = os.environ.get(variable)
    if flavor == "auto":
        os.environ.pop(variable, None)
    else:
        os.environ[variable] = flavor
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(variable, None)
        else:
            os.environ[variable] = previous


def tensor_metadata(tensor):
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "storage_offset": tensor.storage_offset(),
    }


def run_benchmark(script, cases, make_case, variable, controls=DEFAULT_CONTROLS):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument(
        "--case", action="append", choices=[case["name"] for case in cases]
    )
    parser.add_argument(
        "--dtypes",
        nargs="+",
        default=["float32", "float16", "bfloat16"],
        choices=["float32", "float16", "bfloat16", "int32", "int64"],
    )
    parser.add_argument(
        "--role",
        choices=["baseline", "candidate"],
        default="candidate",
        help="Baseline measures auto only, including current generic 3D ternary dispatch.",
    )
    parser.add_argument(
        "--controls",
        nargs="+",
        default=["auto", *controls],
        choices=["auto", "scalar", *controls],
    )
    parser.add_argument("--label")
    parser.add_argument(
        "--expected-sha",
        help="Exact 40-digit torch.version.git_version; required to run.",
    )
    parser.add_argument(
        "--patch-sha",
        help="Optional SHA256 of an uncommitted source patch; a label, not build verification.",
    )
    parser.add_argument("--batch", type=int, default=100)
    parser.add_argument("--min-run-time", type=float, default=0.2)
    args = parser.parse_args()
    selected = [case for case in cases if not args.case or case["name"] in args.case]
    if args.list_cases:
        print(json.dumps(selected, indent=2))
        return
    if (
        not args.label
        or not args.expected_sha
        or not re.fullmatch(r"[0-9a-fA-F]{40}", args.expected_sha)
    ):
        parser.error("--label and exact --expected-sha are required")
    if args.patch_sha and not re.fullmatch(r"[0-9a-fA-F]{64}", args.patch_sha):
        parser.error("--patch-sha must be a SHA256 digest")
    if args.batch < 1 or args.min_run_time <= 0:
        parser.error("--batch and --min-run-time must be positive")

    # Keep --help/--list-cases usable on review machines without PyTorch.
    import torch
    from torch.utils.benchmark import Timer

    if torch.version.git_version != args.expected_sha.lower():
        raise RuntimeError(
            f"Loaded torch SHA {torch.version.git_version}, expected {args.expected_sha}"
        )
    if not torch.backends.mps.is_available():
        raise RuntimeError("An MPS-enabled build and device are required")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise RuntimeError("Disable PYTORCH_ENABLE_MPS_FALLBACK for device-only timing")
    controls = ["auto"] if args.role == "baseline" else args.controls
    sources = [Path(script).resolve(), Path(__file__).resolve()]
    provenance = {
        "kind": "provenance",
        "label": args.label,
        "role": args.role,
        "torch_sha": torch.version.git_version,
        "torch_version": torch.__version__,
        "torch_file": str(Path(torch.__file__).resolve()),
        "python": sys.executable,
        "source_patch_sha256_label": args.patch_sha,
        "scripts_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources
        },
        "platform": platform.platform(),
        "machine": platform.machine(),
        "gpu": json.loads(
            subprocess.check_output(
                ["system_profiler", "SPDisplaysDataType", "-json"], text=True
            )
        ),
        "recommended_max_memory_bytes": torch.mps.recommended_max_memory(),
        "warmup_calls": 30,
        "trials": 5,
        "batch_calls": args.batch,
        "min_run_time_seconds": args.min_run_time,
        "force_variable": variable,
        "controls": controls,
        "environment": {
            key: os.environ.get(key)
            for key in (
                "PYTORCH_ENABLE_MPS_FALLBACK",
                "PYTORCH_MPS_FAST_MATH",
                "PYTORCH_MPS_PREFER_METAL",
            )
        },
        "control_note": "Requested flavor; structural guards and missing registrations may fall back. No route trace collected.",
        "reference": "CPU operation on the same rounded input values, widened to float32 for floating-point arithmetic.",
    }
    print(json.dumps(provenance), flush=True)

    def synchronized_clock():
        torch.mps.synchronize()
        return time.perf_counter()

    order = random.Random(0)
    for case in selected:
        for dtype_name in case.get("dtypes", args.dtypes):
            if case.get("floating_only") and dtype_name.startswith("int"):
                continue
            torch.manual_seed(0)
            fn, reference, inputs = make_case(torch, case, getattr(torch, dtype_name))
            flavors = list(controls)
            order.shuffle(flavors)
            for flavor in flavors:
                with force_flavor(variable, flavor):
                    result = fn()
                    output_metadata = tensor_metadata(result)
                    actual = result.cpu()
                    del result
                    tolerances = {
                        torch.float16: (1e-3, 1e-3),
                        torch.bfloat16: (1e-2, 1e-2),
                        torch.float32: (1e-4, 1e-5),
                    }
                    rtol, atol = tolerances.get(reference.dtype, (0, 0))
                    torch.testing.assert_close(actual, reference, rtol=rtol, atol=atol)
                    error = (
                        actual.to(torch.float64) - reference.to(torch.float64)
                    ).abs()
                    for _ in range(30):
                        fn()
                    torch.mps.synchronize()
                    timer = Timer(
                        stmt="for _ in range(batch): fn()",
                        globals={"fn": fn, "batch": args.batch},
                        timer=synchronized_clock,
                    )
                    measurements = [
                        timer.blocked_autorange(min_run_time=args.min_run_time)
                        for _ in range(5)
                    ]
                    per_call_us = [
                        measurement.mean * 1e6 / args.batch
                        for measurement in measurements
                    ]
                    print(
                        json.dumps(
                            {
                                "kind": "timing",
                                "label": args.label,
                                "torch_sha": torch.version.git_version,
                                "case": case,
                                "dtype": dtype_name,
                                "requested_flavor": flavor,
                                "inputs": inputs,
                                "output": output_metadata,
                                "reference_passed": True,
                                "max_abs_error": error.max().item()
                                if error.numel()
                                else 0,
                                "rtol": rtol,
                                "atol": atol,
                                "trial_mean_us_per_call": per_call_us,
                                "mean_us_per_call": statistics.mean(per_call_us),
                                "raw_trials": [
                                    {
                                        "raw_times_seconds": measurement.raw_times,
                                        "number_per_run": measurement.number_per_run,
                                        "calls_per_statement": args.batch,
                                    }
                                    for measurement in measurements
                                ],
                            }
                        ),
                        flush=True,
                    )
            del fn, reference, inputs
            torch.mps.empty_cache()
