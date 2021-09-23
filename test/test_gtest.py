import subprocess
import time
import os
from pathlib import Path
from typing import List
from torch.testing._internal.common_utils import (
    TestCase,
    run_tests,
    TEST_SAVE_XML,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_BINARY_DIR = REPO_ROOT / "build" / "bin"
BUILD_ENVIRONMENT = os.getenv("BUILD_ENVIRONMENT", "")

# This is a temporary list of tests that use this framework rather than get run
# as regular binaries.
# TODO: Once all the C++ tests have been migrated to this wrapper, we can delete
# this list
ALLOWLISTED_TEST = {
    "test_jit",
}


def run_binary(binary: Path, test_name: str, extra_flags: List[str] = None):
    cmd = [str(binary)]
    if extra_flags is None:
        extra_flags = []
    print(TEST_SAVE_XML)

    if TEST_SAVE_XML is not None:
        # Suffix copied from XMLTestRunner.__init__
        suffix = time.strftime("%Y%m%d%H%M%S")

        xml_path = (
            Path(TEST_SAVE_XML) / "test.test_gtest" / f"TEST-{test_name}-{suffix}.xml"
        )
        cmd += [f"--gtest_output=xml:{xml_path}"]
    cmd += extra_flags
    print(f"[gtest runner] {' '.join(cmd)}")
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"C++ test '{binary.name}' failed ({binary})")


class GTest(TestCase):
    def test_jit(self, binary: Path, test_name: str):
        if "cuda" in BUILD_ENVIRONMENT:
            run_binary(binary, test_name)
        else:
            run_binary(binary, test_name, extra_flags=["--gtest_filter=-*CUDA"])


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
