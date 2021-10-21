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
    IS_IN_CI,
    skipIfRocm,
    TEST_WITH_ROCM,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_BINARY_DIR = REPO_ROOT / "build" / "bin"

env = os.environ.copy()

if IS_MACOS:
    env["DYLD_LIBRARY_PATH"] = str(Path(torch.__file__).resolve().parent / "lib")

if IS_WINDOWS:
    # Windows needs these paths to find the relevant DLLs to run the test
    # binaries. Some binaries only need c10.dll, others need more which you can
    # find by using 'dumpbin.exe <binary>'
    # libiomp5.dll is also necessary for a lot of them (find via 'dir *lib*omp*dll /s /p')
    # Remote desktop will also show a popup of missing DLLs when running a test
    # binary which doesn't show up when running from cmd/bash/powershell.
    paths = [
        r"C:\Jenkins\Miniconda3\Library\bin",
        r"C:\actions-runner\_work\pytorch\pytorch\build\bin",
        r"C:\Jenkins\Miniconda3",
        r"C:\actions-runner\bin",
    ]
    env["PATH"] = ";".join(paths) + ";" + env["PATH"]

if IS_IN_CI:
    if IS_MACOS:
        TEST_BINARY_DIR = Path(torch.__file__).resolve().parent / "bin"
    elif IS_WINDOWS:
        TEST_BINARY_DIR = REPO_ROOT / "build"
BUILD_ENVIRONMENT = os.getenv("BUILD_ENVIRONMENT", "")

print(f"[remove] USING PATH {TEST_BINARY_DIR}")

# This is a temporary list of tests that use this framework rather than get run
# as regular binaries.
# TODO: Once all the C++ tests have been migrated to this wrapper, we can delete
# this list
ALLOWLISTED_TESTS = {
    "test_jit",
}


def run_cmd(cmd: List[str]) -> Any:
    print(f"[gtest runner] {' '.join(cmd)}")
    proc = subprocess.run(cmd, env=env)
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("main")
        if TEST_WITH_ROCM:
            # C++ tests don't run on rocm
            return

        if not TEST_BINARY_DIR.exists():
            raise RuntimeError(
                f"{TEST_BINARY_DIR} does not exist, this test "
                "must run from a PyTorch checkout"
            )

        print("CWD", os. getcwd())
        print(TEST_BINARY_DIR)
        print("globs")
        print('list(TEST_BINARY_DIR.glob("*"))', list(TEST_BINARY_DIR.glob("*")))
        print(
            'list(TEST_BINARY_DIR.glob("*test*"))', list(TEST_BINARY_DIR.glob("*test*"))
        )
        other_bin_dir = REPO_ROOT / "build" / "bin"
        print(other_bin_dir, other_bin_dir.exists())
        print(other_bin_dir.glob("*"))
        self.binaries = {}

        for binary in TEST_BINARY_DIR.glob("*test*"):
            print("adding", binary)
            # If the test already has a properly formatted name, don't prepend a
            # redundant 'test_'
            if IS_WINDOWS and binary.suffix != ".exe":
                continue

            if binary.name.startswith("test_"):
                if IS_WINDOWS:
                    # Get rid of the ".exe"
                    test_name = binary.stem
                else:
                    test_name = binary.name
            else:
                test_name = f"test_{binary.name}"

            if test_name not in ALLOWLISTED_TESTS:
                continue

            self.binaries[test_name] = binary

            if getattr(self, test_name, None) is None:
                # Add the test case if it's not hardcoded
                def test_case(self):
                    run_binary(binary, test_name)

                setattr(self, test_name, test_case)

            # maybe_existing_case = generate_test_case(
            #     getattr(GTest, test_name, None), binary, test_name
            # )
            # setattr(
            #     GTest, test_name, maybe_existing_case,
            # )

    @skipIfRocm
    def test_jit(self):
        test_name = "test_jit"
        binary = self.binaries[test_name]
        setup_path = REPO_ROOT / "test" / "cpp" / "jit" / "tests_setup.py"
        # These tests fail on windows only (this wasn't caught before switching
        # to the Python runner since test_jit.exe wasn't run during windows
        # testing)
        windows_failing_tests = [
            "BackendTest.ToBackend",
            "BackendTest.ToBackendNotAvailable",
            "BackendTest.TestCompiler",
            "BackendTest.TestComposite",
            "BackendTest.TestCompositeWithSetStates",
            "BackendTest.TestConsistencyOfCompositeWithSetStates",
            "BackendTest.TestCompilerNotSupport",
            "BackendTestDebugInfo.TestCompiler",
            "BackendTestDebugInfo.TestExceptionStackForCompilerWithModuleHierarchy",
            "BackendTestDebugInfo.TestExceptionStackForCompilerWithTwoLevelModuleHierarchy",
            "BackendTestDebugInfo.TestExceptionStackForCompilerWithLoweredSubModule",
            "BackendTestDebugInfo.TestExceptionStackForCompilerWithSelectiveLoweredSubModule",
            "ClassTypeTest.IdenticalTypesDifferentCus",
            "LiteInterpreterTest.BackPortByteCodeModelAllVersions",
            "LiteInterpreterTest.isCompatibleSuccess",
            "LiteInterpreterTest.isCompatibleFail",
            "MobileTypeParserTest.NonIdentifierRaises",
            "JitLoggingTest.CheckOutputStreamSetting",
        ]

        def gtest_filter():
            filter = "*"
            exclusions = []

            if "cuda" in BUILD_ENVIRONMENT:
                exclusions.append("*CUDA")

            if IS_WINDOWS:
                exclusions += windows_failing_tests

            if len(exclusions) > 0:
                filter += "-" + ":".join(exclusions)

            return f"--gtest_filter={filter}"

        flags = []
        filter = gtest_filter()
        if filter != "--gtest_filter=*":
            flags = [filter]

        run_cmd([sys.executable, str(setup_path), "setup"])
        run_binary(binary, test_name, extra_flags=flags)
        run_cmd([sys.executable, str(setup_path), "shutdown"])


# def generate_test_case(existing_case, binary: Path, test_name: str):
#     if existing_case is None:

#         def test_case(self):
#             run_binary(binary, test_name)

#     else:

#         def test_case(self):
#             existing_case(self, binary, test_name)

#     return skipIfRocm(test_case)


print("not main")
if __name__ == "__main__":
    # print("main")
    # if not TEST_BINARY_DIR.exists():
    #     print(
    #         f"{TEST_BINARY_DIR} does not exist, this test "
    #         "must run from a PyTorch checkout"
    #     )
    #     exit(1)

    # print(TEST_BINARY_DIR)
    # print("globs")
    # print('list(TEST_BINARY_DIR.glob("*"))', list(TEST_BINARY_DIR.glob("*")))
    # print('list(TEST_BINARY_DIR.glob("*test*"))', list(TEST_BINARY_DIR.glob("*test*")))
    # for binary in TEST_BINARY_DIR.glob("*test*"):
    #     # If the test already has a properly formatted name, don't prepend a
    #     # redundant 'test_'
    #     if IS_WINDOWS and binary.suffix != ".exe":
    #         continue

    #     if binary.name.startswith("test_"):
    #         if IS_WINDOWS:
    #             # Get rid of the ".exe"
    #             test_name = binary.stem
    #         else:
    #             test_name = binary.name
    #     else:
    #         test_name = f"test_{binary.name}"

    #     if test_name not in ALLOWLISTED_TESTS:
    #         continue

    #     maybe_existing_case = generate_test_case(
    #         getattr(GTest, test_name, None), binary, test_name
    #     )
    #     setattr(
    #         GTest, test_name, maybe_existing_case,
    #     )

    # Don't 'save_xml' since gtest does that for us and we don't want to
    # duplicate it for these test cases
    run_tests(save_xml=False)
