#!/usr/bin/env python3
"""Map test files to CODEOWNERS module labels, else their review surface."""

import json
import subprocess
import sys
from pathlib import Path

from check_taxonomy import (
    compile_glob,
    parse_patterns,
    pattern_specificity,
    TaxonomyIndex,
    tracked_paths,
)


IGNORED_OWNERS = {"unknown", "tests"}
EXCLUDED_DIRECTORIES = {"fb", "third_party"}
TEST_GLOBS = [
    # Python tests.
    "test/**/test_*.py",
    "test/**/*_test.py",
    "test/distributed/**/*_example.py",
    # Binary validation tests.
    ".ci/pytorch/test_example_code/**/*.cpp",
    # Native tests.
    "test/**/*.cpp",
    "test/**/*.cc",
    "test/**/*.cxx",
    "test/**/*.cu",
    "test/**/*.h",
    "test/**/*.mm",
    # Android tests.
    "android/pytorch_android/src/androidTest/cpp/**/*.cpp",
    # ATen tests.
    "aten/src/ATen/**/test/**/*.cpp",
    "aten/src/ATen/**/test/**/*.cc",
    "aten/src/ATen/**/test/**/*.cu",
    "aten/src/ATen/**/test/**/*.h",
    "aten/src/ATen/**/test/**/*.mm",
    # MPS tests.
    "aten/src/ATen/native/metal/mpscnn/tests/**/*.h",
    "aten/src/ATen/native/metal/mpscnn/tests/**/*.mm",
    # Static runtime tests.
    "benchmarks/static_runtime/**/*.cc",
    "benchmarks/static_runtime/**/*.h",
    # c10 tests.
    "c10/**/test/**/*.cpp",
    "c10/**/test/**/*.cu",
    "c10/**/test/**/*.h",
    # Tests alongside implementation code.
    "aten/src/ATen/core/**/*_test.cpp",
    "caffe2/**/*_test.cc",
]


def collect_test_owners(repo: Path) -> dict[str, list[str]]:
    """Map every tracked test file to CODEOWNERS module labels or its review surface."""
    index = TaxonomyIndex(parse_patterns(repo / "CODEOWNERS")[0])
    test_globs = [compile_glob(test_glob) for test_glob in TEST_GLOBS]
    owners_by_file = {}
    for path in tracked_paths(repo):
        if EXCLUDED_DIRECTORIES.intersection(path.split("/")) or not any(
            test_glob.fullmatch(path) for test_glob in test_globs
        ):
            continue
        owners = set()
        matches = index.matches(path)
        if matches:
            pattern = max(matches, key=pattern_specificity)
            owners = {
                label.removeprefix("module: ")
                for label in pattern.labels
                if label.startswith("module: ")
            } - IGNORED_OWNERS
            if not owners:
                owners = {pattern.group}
        owners_by_file[path] = sorted(owners)
    if not owners_by_file:
        raise ValueError(f"No test files found in {repo}")
    return owners_by_file


def main() -> int:
    output = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True)
    print(json.dumps(collect_test_owners(Path(output.strip())), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
