import subprocess
import time
import os
import sys
from pathlib import Path
from typing import List, Any
import torch
from torch.testing._internal.common_utils import (
    TestCase,
    run_tests,
    TEST_SAVE_XML,
    IS_WINDOWS,
    IS_MACOS,
    IS_IN_CI
)

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_BINARY_DIR = REPO_ROOT / "build" / "bin"

if IS_IN_CI:
    if IS_WINDOWS:
        TEST_BINARY_DIR = REPO_ROOT / "torch" / "bin"
    elif IS_MACOS:
        # maybe have to set DYLD_LIBRARY_PATH to .parent / "lib"
        TEST_BINARY_DIR = Path(torch.__file__).resolve().parent / "bin"
        # TEST_BINARY_DIR = REPO_ROOT.parent / "cpp-build" / "bin"
BUILD_ENVIRONMENT = os.getenv("BUILD_ENVIRONMENT", "")

print(f"[remove] USING PATH {TEST_BINARY_DIR}")

# This is a temporary list of tests that use this framework rather than get run
# as regular binaries.
# TODO: Once all the C++ tests have been migrated to this wrapper, we can delete
# this list
ALLOWLISTED_TEST = {
    "test_jit",
}


def run_cmd(cmd: List[str]) -> Any:
    print(f"[gtest runner] {' '.join(cmd)}")
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"Command '{cmd}' failed")
    return proc


def run_binary(binary: Path, test_name: str, extra_flags: List[str] = None):
    cmd = [str(binary)]
    if extra_flags is None:
        extra_flags = []

    if TEST_SAVE_XML is not None:
        # Suffix copied from XMLTestRunner.__init__
        suffix = time.strftime("%Y%m%d%H%M%S")

        xml_path = (
            Path(TEST_SAVE_XML) / "test.test_gtest" / f"TEST-{test_name}-{suffix}.xml"
        )
        cmd += [f"--gtest_output=xml:{xml_path}"]
    cmd += extra_flags

    run_cmd(cmd)


class GTest(TestCase):
    """
    This class has methods added to it below for each C++ test in
    build/bin/*test*. Wrapping tests in Python makes it easier to ensure we are
    running tests in a consistent way and have the C++ tests participate in
    sharding.

    To add a custom test that does more than just run the test binary, add a
    method to this class named the same as the test binary (or test_<name> if
    the binary's name doesn't start with 'test_')
    """

    def test_jit(self, binary: Path, test_name: str):
        setup_path = REPO_ROOT / "test" / "cpp" / "jit" / "tests_setup.py"

        run_cmd([sys.executable, str(setup_path), "setup"])
        if "cuda" in BUILD_ENVIRONMENT:
            run_binary(binary, test_name)
        else:
            run_binary(binary, test_name, extra_flags=["--gtest_filter=-*CUDA"])
        run_cmd([sys.executable, str(setup_path), "shutdown"])


def generate_test_case(existing_case, binary: Path, test_name: str):
    if existing_case is None:

        def test_case(self):
            run_binary(binary, test_name)

    else:

        def test_case(self):
            existing_case(self, binary, test_name)

    return test_case


if __name__ == "__main__":
    if not TEST_BINARY_DIR.exists():
        print(
            f"{TEST_BINARY_DIR} does not exist, this test "
            "must run from a PyTorch checkout"
        )
        exit(1)

    for binary in TEST_BINARY_DIR.glob("*test*"):
        # If the test already has a properly formatted name, don't prepend a
        # redundant 'test_'
        if binary.name.startswith("test_"):
            test_name = binary.name
        else:
            test_name = f"test_{binary.name}"

        if test_name not in ALLOWLISTED_TEST:
            continue

        maybe_existing_case = generate_test_case(
            getattr(GTest, test_name, None), binary, test_name
        )
        setattr(
            GTest,
            test_name,
            maybe_existing_case,
        )

    # Don't 'save_xml' since gtest does that for us and we don't want to
    # duplicate it for these test cases
    run_tests(save_xml=False)
