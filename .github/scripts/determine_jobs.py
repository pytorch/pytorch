#!/usr/bin/env python3
"""Determine which CI jobs to run based on changed files.

This script analyzes the files changed in a PR and outputs which job
categories should run. It's used by job-filter.yml to skip irrelevant
build environments and test configs.

The approach is conservative: when in doubt, run everything. Only
well-understood file patterns trigger job skipping.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path


# Categories of changes and which jobs they require.
# A change is classified into one or more categories based on file path patterns.
# Jobs run if ANY of their required categories match ANY changed file.

# fmt: off
CATEGORY_PATTERNS: dict[str, list[str]] = {
    # Python source changes (non-C++)
    "python": [
        r"^torch/.*\.py$",
        r"^torch/.*\.pyi$",
    ],
    # C++ source changes
    "cpp": [
        r"^aten/",
        r"^c10/",
        r"^torch/csrc/",
        r"^torch/nativert/",
        r"^caffe2/",
        r"^torch/headeronly/",
        r"^build_variables\.bzl$",
    ],
    # CUDA/GPU-specific changes
    "cuda": [
        r"^aten/src/ATen/native/cuda/",
        r"^aten/src/ATen/cuda/",
        r"^c10/cuda/",
        r"^torch/csrc/cuda/",
        r"^torch/cuda/",
    ],
    # ROCm/HIP changes
    "rocm": [
        r"^aten/src/ATen/native/hip/",
        r"^aten/src/ATen/hip/",
        r"^c10/hip/",
        r"\.hip$",
    ],
    # XPU changes
    "xpu": [
        r"^c10/xpu/",
        r"^aten/src/ATen/native/xpu/",
        r"^torch/xpu/",
    ],
    # MPS/Apple changes
    "mps": [
        r"^aten/src/ATen/native/mps/",
        r"^aten/src/ATen/mps/",
        r"^c10/metal/",
        r"\.mm$",
        r"\.metal$",
    ],
    # Dynamo changes
    "dynamo": [
        r"^torch/_dynamo/",
    ],
    # Inductor changes
    "inductor": [
        r"^torch/_inductor/",
    ],
    # Export changes
    "export": [
        r"^torch/export/",
        r"^torch/_export/",
    ],
    # Distributed changes
    "distributed": [
        r"^torch/distributed/",
        r"^torch/csrc/distributed/",
    ],
    # JIT changes
    "jit": [
        r"^torch/jit/",
        r"^torch/csrc/jit/",
    ],
    # ONNX changes
    "onnx": [
        r"^torch/onnx/",
        r"^torch/csrc/onnx/",
    ],
    # Build system changes
    "build": [
        r"^cmake/",
        r"CMakeLists\.txt$",
        r"^setup\.py$",
        r"^pyproject\.toml$",
        r"^requirements\.txt$",
    ],
    # CI infrastructure changes
    "ci": [
        r"^\.github/",
        r"^\.ci/",
        r"^\.circleci/",
        r"^\.jenkins/",
    ],
    # Documentation only
    "docs": [
        r"^docs/",
        r"^README\.md$",
        r"^CODEOWNERS$",
    ],
    # Test infrastructure
    "test_infra": [
        r"^test/",
        r"^tools/testing/",
        r"^torch/testing/",
    ],
    # Codegen
    "codegen": [
        r"^torchgen/",
        r"^aten/src/ATen/native/native_functions\.yaml$",
        r"^tools/autograd/",
    ],
    # Functorch
    "functorch": [
        r"^torch/_functorch/",
        r"^functorch/",
    ],
    # Benchmarks
    "benchmarks": [
        r"^benchmarks/",
    ],
    # Mobile
    "mobile": [
        r"^android/",
        r"^ios/",
    ],
    # Quantization
    "quantization": [
        r"^torch/ao/",
        r"^torch/quantization/",
    ],
    # OpenReg (custom device registration)
    "openreg": [
        r"^test/cpp_extensions/open_registration_extension/",
    ],
}
# fmt: on


# Which job display names should run for each category.
# "always" jobs run regardless of what changed.
# If a category is not listed, it triggers all jobs (conservative).
CATEGORY_TO_JOBS: dict[str, list[str]] = {
    "docs": [
        "linux-jammy-py3.10-gcc11",  # needed for linux-docs build dep
        "linux-docs",
    ],
    "ci": [
        # CI changes need at least one default build+test to validate
        "linux-jammy-py3.10-gcc11",
    ],
}

# Jobs that always run regardless of what changed
ALWAYS_RUN_JOBS = [
    "linux-jammy-py3.10-gcc11",  # primary CPU build + default/distributed/special configs
]

# Jobs that only run when specific categories are present
CONDITIONAL_JOBS: dict[str, list[str]] = {
    # ASAN: only for C++/build/codegen changes (catches memory errors in compiled code)
    "linux-jammy-py3.10-clang18-asan": ["cpp", "build", "codegen", "cuda"],
    # ONNX: only for onnx/export changes
    "linux-jammy-py3.10-clang15-onnx": ["onnx", "export", "cpp", "codegen", "build"],
    # Build-only jobs: only for C++/build changes
    "linux-jammy-py3.10-gcc11-no-ops": ["cpp", "build", "codegen"],
    "linux-jammy-py3.10-gcc11-pch": ["cpp", "build", "codegen"],
    "linux-jammy-py3.10-gcc11-mobile-lightweight-dispatch-build": [
        "cpp",
        "build",
        "mobile",
    ],
    # CUDA builds: only for C++/CUDA/build/codegen changes
    "linux-jammy-cuda12.8-cudnn9-py3.10-clang15": [
        "cpp",
        "cuda",
        "build",
        "codegen",
    ],
    # CUDA inductor benchmarks: for inductor/dynamo/functorch/codegen changes
    "cuda12.8-py3.10-gcc11-sm75": [
        "inductor",
        "dynamo",
        "functorch",
        "codegen",
        "cpp",
        "build",
        "benchmarks",
    ],
    "cuda13.0-py3.10-gcc11-sm75": [
        "inductor",
        "dynamo",
        "functorch",
        "codegen",
        "cpp",
        "build",
        "benchmarks",
    ],
    # ROCm: only for C++/CUDA/ROCm/build changes
    "linux-jammy-rocm-py3.10": ["cpp", "cuda", "rocm", "build", "codegen"],
    # XPU: only for XPU/C++/build changes
    "linux-jammy-xpu-n-py3.10": ["xpu", "cpp", "build", "codegen"],
    # Bazel: only for build system changes
    "linux-jammy-cpu-py3.10-gcc11-bazel-test": ["build", "cpp", "codegen"],
}


def get_changed_files() -> list[str]:
    """Get changed files from environment or git."""
    # In CI, changed files can be provided via env var
    files_str = os.environ.get("CHANGED_FILES", "")
    if files_str:
        return [f.strip() for f in files_str.split("\n") if f.strip()]

    # For PRs, diff against base
    base_sha = os.environ.get("BASE_SHA", "")
    head_sha = os.environ.get("HEAD_SHA", "HEAD")

    if base_sha:
        cmd = ["git", "diff", "--name-only", base_sha, head_sha]
    else:
        cmd = ["git", "diff", "--name-only", "HEAD~1", "HEAD"]

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        print(f"Warning: git diff failed: {result.stderr}", file=sys.stderr)
        return []

    return [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]


def classify_changes(changed_files: list[str]) -> set[str]:
    """Classify changed files into categories."""
    categories: set[str] = set()

    for filepath in changed_files:
        matched = False
        for category, patterns in CATEGORY_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, filepath):
                    categories.add(category)
                    matched = True
                    break
        if not matched:
            # Unknown file type — be conservative, trigger everything
            categories.add("unknown")

    return categories


def determine_jobs(categories: set[str]) -> list[str]:
    """Determine which jobs should run based on change categories."""
    # If any category is unknown or not in our mapping, run everything
    if "unknown" in categories:
        return []  # empty = run all

    # If there are only docs/CI changes, use restricted job lists
    docs_ci_only = categories.issubset({"docs", "ci"})
    if docs_ci_only:
        jobs: set[str] = set()
        for cat in categories:
            if cat in CATEGORY_TO_JOBS:
                jobs.update(CATEGORY_TO_JOBS[cat])
        return sorted(jobs) if jobs else []

    # Start with always-run jobs
    jobs_to_run: set[str] = set(ALWAYS_RUN_JOBS)

    # Add conditional jobs if their categories are triggered
    for job_name, required_categories in CONDITIONAL_JOBS.items():
        if categories & set(required_categories):
            jobs_to_run.add(job_name)

    # Python-only changes (no C++/build/codegen) get a reduced set.
    # We still need multiple Python version builds for compatibility testing.
    has_compiled_changes = bool(
        categories & {"cpp", "build", "codegen", "cuda", "rocm", "xpu", "mps"}
    )

    if not has_compiled_changes:
        # For pure Python changes, run:
        # - Primary build (gcc11) — always
        # - One additional Python version for compat
        # - Inductor benchmarks if dynamo/inductor/functorch changed
        jobs_to_run.add("linux-jammy-py3.14t-clang15")
        # Skip: ASAN, ONNX, no-ops, pch, mobile, CUDA build, bazel, ROCm, XPU
        # These are already excluded because they require compiled change categories
    else:
        # Compiled changes need full build coverage
        jobs_to_run.add("linux-jammy-py3.14t-clang15")
        jobs_to_run.add("linux-jammy-py3.10-clang15")
        jobs_to_run.add("linux-jammy-py3.14-clang15")

    # Docs build
    if "docs" in categories or not docs_ci_only:
        jobs_to_run.add("linux-docs")

    return sorted(jobs_to_run)


def main() -> None:
    changed_files = get_changed_files()

    if not changed_files:
        print("No changed files detected, running all jobs", file=sys.stderr)
        print(json.dumps({"jobs": "", "categories": [], "changed_files": []}))
        return

    categories = classify_changes(changed_files)
    jobs = determine_jobs(categories)

    print(f"Changed files ({len(changed_files)}):", file=sys.stderr)
    for f in changed_files[:20]:
        print(f"  {f}", file=sys.stderr)
    if len(changed_files) > 20:
        print(f"  ... and {len(changed_files) - 20} more", file=sys.stderr)
    print(f"Categories: {sorted(categories)}", file=sys.stderr)
    print(f"Jobs to run: {jobs if jobs else 'ALL'}", file=sys.stderr)

    # Output as space-padded string matching job-filter.yml format
    if jobs:
        jobs_str = " " + " ".join(jobs) + " "
    else:
        jobs_str = ""

    print(json.dumps({
        "jobs": jobs_str,
        "categories": sorted(categories),
        "changed_files": changed_files,
    }))


if __name__ == "__main__":
    main()
