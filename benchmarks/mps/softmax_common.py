"""Shared model shapes and result metadata for the MPS softmax benchmarks."""

import hashlib
import json
import platform
import subprocess
from pathlib import Path

import torch


CASES = [
    ((4, 499, 46), 2),
    ((1, 201088), 1),
    ((5, 51866), 1),
    ((1, 151936), 1),
    ((64, 1000), 1),
    ((4, 129280), 1),
    ((4, 151936), 1),
    ((4096, 30522), 1),
    ((2, 150, 512, 512), 1),
    ((8, 154880), 1),
    ((4, 499, 2341), 1),
    ((1024, 50257), 1),
    ((2048, 128256), 1),
    ((2048, 129280), 1),
    ((2048, 151936), 1),
    ((8192, 151936), 1),
    ((2048, 154880), 1),
    ((1024, 201088), 1),
    ((4, 1023, 151936), 2),
    ((4, 151936, 128), 1),
]


def add_arguments(parser):
    parser.add_argument(
        "--op", choices=("softmax", "log_softmax", "both"), default="both"
    )
    parser.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16"), default="float32"
    )
    parser.add_argument("--cases", type=int, nargs="+", choices=range(len(CASES)))
    parser.add_argument(
        "--output", type=Path, help="Save metadata and measurements as JSON."
    )


def selected_cases(args):
    return [
        (i, *CASES[i])
        for i in (range(len(CASES)) if args.cases is None else args.cases)
    ]


def operators(args):
    return ("softmax", "log_softmax") if args.op == "both" else (args.op,)


def save_results(args, measurements):
    if args.output is None:
        return
    kernel = Path("aten/src/ATen/native/mps/kernels/SoftMax.metal")
    data = {
        "build_revision": torch.version.git_version,
        "git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], text=True),
        "kernel_sha256": hashlib.sha256(kernel.read_bytes()).hexdigest(),
        "platform": platform.platform(),
        "torch_path": torch.__file__,
        "options": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "measurements": measurements,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + "\n")
