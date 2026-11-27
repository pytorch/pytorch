"""Which changed paths are build, dependency or test-infrastructure paths.

A change touching one gets the top of the TD scorer's budget window
(scorer_model.budget_share). Standard library only and free of repo imports:
pytorch/test-infra loads this file by path to check its own copy of the lists.
"""

from __future__ import annotations


MAX_CHANGED_FILES = 3000

BUILD_DIRS = (
    "third_party/",
    "cmake/",
    ".ci/",
    ".github/",
    "torchgen/",
    "tools/setup_helpers/",
    "tools/autograd/",
    "tools/testing/",
    "tools/stats/",
    "torch/testing/_internal/",
    "test/dynamo_expected_failures/",
    "test/dynamo_skips/",
    "test/inductor_expected_failures/",
    "test/inductor_skips/",
    "aten/src/ATen/templates/",
    "tools/jit/",
    "tools/pyi/",
    "tools/amd_build/",
)
BUILD_FILES = ("test/run_test.py",)
BUILD_BASENAMES = (
    "CMakeLists.txt",
    "setup.py",
    "setup.cfg",
    "pyproject.toml",
    "version.txt",
    "BUILD.bazel",
    "BUILD",
    "WORKSPACE",
    "conftest.py",
    "pytest.ini",
    "native_functions.yaml",
    "tags.yaml",
    "MODULE.bazel",
    ".bazelrc",
    ".bazelversion",
)
BUILD_BASENAME_PREFIXES = ("requirements",)
BUILD_SUFFIXES = (".cmake", ".bzl")


def is_build_path(path: str) -> bool:
    """Whether a changed path is a build, dependency or test-infrastructure path."""
    base = path.rsplit("/", 1)[-1]
    return (
        path.startswith(BUILD_DIRS)
        or path in BUILD_FILES
        or base in BUILD_BASENAMES
        or (base.startswith(BUILD_BASENAME_PREFIXES) and base.endswith(".txt"))
        or base.endswith(BUILD_SUFFIXES)
    )


def count_build_files(paths: list[str]) -> int | None:
    """How many changed paths are build paths; None when no path is listed.

    Empty paths are dropped first: git lists an empty diff as [""]. GitHub
    truncates the changed-files list at MAX_CHANGED_FILES, so a list that long
    counts as all build paths: the unlisted files could be any.
    """
    paths = [p for p in paths if p]
    if not paths:
        return None
    if len(paths) >= MAX_CHANGED_FILES:
        return len(paths)
    return sum(is_build_path(p) for p in paths)
