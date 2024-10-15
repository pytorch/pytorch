import os
import time

from ..tool import clang_coverage, gcc_coverage
from ..util.setting import TestList, TestPlatform
from ..util.utils import get_raw_profiles_folder, print_time
from .utils import get_oss_binary_file


def clang_run(tests: TestList) -> None:
    start_time = time.time()
    for test in tests:
        # raw_file
        raw_file = os.path.join(get_raw_profiles_folder(), test.name + ".profraw")
        # binary file
        binary_file = get_oss_binary_file(test.name, test.test_type)
        clang_coverage.run_target(
            binary_file, raw_file, test.test_type, TestPlatform.OSS
        )
    print_time("running binaries takes time: ", start_time, summary_time=True)


def gcc_run(tests: TestList) -> None:
    start_time = time.time()
    for test in tests:
        # binary file
        binary_file = get_oss_binary_file(test.name, test.test_type)
        gcc_coverage.run_target(binary_file, test.test_type)
    print_time("run binaries takes time: ", start_time, summary_time=True)
