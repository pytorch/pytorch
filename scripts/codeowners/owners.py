#!/usr/bin/env python3
"""Map test files to owners: `# Owner(s):` labels, else the CODEOWNERS surface."""

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


OWNERS_PREFIXES = ("# Owner(s): ", "// Owner(s): ")
IGNORED_OWNERS = {"unknown", "tests"}
EXCLUDED_PATHS = {"test/run_test.py"}
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


def header_owners(test_path: Path, relative_path: str) -> set[str]:
    """Return the `module:` labels of the file's `# Owner(s):` header, if any."""
    with test_path.open(encoding="utf-8") as test_file:
        for line in test_file:
            if not line.startswith(OWNERS_PREFIXES):
                continue
            try:
                labels = json.loads(line.partition(": ")[2])
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid owners header in {relative_path}") from error
            if not isinstance(labels, list) or any(
                not isinstance(label, str) for label in labels
            ):
                raise ValueError(f"Expected a list of owner labels in {relative_path}")
            return {
                label.removeprefix("module: ")
                for label in labels
                if label.startswith("module: ")
            } - IGNORED_OWNERS
    return set()


def collect_test_owners(repo: Path) -> dict[str, list[str]]:
    """Map every tracked test file to its owners, else to its review surface."""
    index = TaxonomyIndex(parse_patterns(repo / "CODEOWNERS")[0])
    test_globs = [compile_glob(test_glob) for test_glob in TEST_GLOBS]
    owners_by_file = {}
    for path in tracked_paths(repo):
        if (
            path in EXCLUDED_PATHS
            or EXCLUDED_DIRECTORIES.intersection(path.split("/"))
            or not any(test_glob.fullmatch(path) for test_glob in test_globs)
        ):
            continue
        owners = header_owners(repo / path, path)
        if not owners:
            matches = index.matches(path)
            if matches:
                owners = {max(matches, key=pattern_specificity).group}
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
