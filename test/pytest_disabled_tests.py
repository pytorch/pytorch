import pytest
import os
import json
from typing import Dict, Optional

from _pytest.config.argparsing import Parser


DEFAULT_DISABLED_TESTS_FILE = ".pytorch-disabled-tests.json"


def pytest_addoptions(parser: Parser):
    """Add options to disable tests via file."""
    group = parser.getgroup("disabled")
    group.addoption(
        "--import-disabled-tests",
        type=str,
        nargs="?",
        const=DEFAULT_DISABLED_TESTS_FILE,
    )


class DisabledTestsPlugin:
    def __init__(self, config):
        self.config = config
        disabled_tests_file = config.getoption("import_disabled_tests")
        if os.path.exists(disabled_tests_file):
            with open(disabled_tests_file) as fp:
                global disabled_tests_dict
                self.disabled_tests_dict = json.load(fp)
                os.environ["DISABLED_TESTS_FILE"] = disabled_tests_file

    def remove_device_and_dtype_suffixes(test_name: str) -> str:
        try:
            # import statement is localized to avoid circular dependency issues with common_device_type.py
            from torch.testing._internal.common_device_type import (
                get_device_type_test_bases,
            )
            from torch.testing._internal.common_dtype import get_all_dtypes

            device_suffixes = [x.device_type for x in get_device_type_test_bases()]
            dtype_suffixes = [str(dt)[len("torch.") :] for dt in get_all_dtypes()]

            test_name_chunks = test_name.split("_")
            if len(test_name_chunks) > 0 and test_name_chunks[-1] in dtype_suffixes:
                if (
                    len(test_name_chunks) > 1
                    and test_name_chunks[-2] in device_suffixes
                ):
                    return "_".join(test_name_chunks[0:-2])
                return "_".join(test_name_chunks[0:-1])
            return test_name
        except:
            return test_name

    def matches_test(self, testname: str, target: str):
        fullname = testname.split(":")[-1]
        classname, testname = fullname.split(".")
        sanitized_testname = self.remove_device_and_dtype_suffixes(testname)

        target_test_parts = target.split()
        if len(target_test_parts) < 2:
            # poorly formed target test name
            return False
        target_testname = target_test_parts[0]
        target_classname = target_test_parts[1][1:-1].split(".")[-1]
        # if test method name or its sanitized version exactly matches the disabled
        # test method name AND allow non-parametrized suite names to disable
        # parametrized ones (TestSuite disables TestSuiteCPU)
        return classname.startswith(target_classname) and (
            target_testname in (testname, sanitized_testname)
        )

    def get_disable_reason(self, testname) -> Optional[str]:
        from torch.testing._internal.common_utils import (
            IS_MACOS,
            IS_WINDOWS,
            IS_LINUX,
            TEST_WITH_ROCM,
            TEST_XPU,
            TEST_WITH_ASAN,
            TEST_WITH_TORCHDYNAMO,
            TEST_WITH_TORCHINDUCTOR,
            TEST_WITH_SLOW,
        )

        for disabled_test, (issue_url, platforms) in self.disabled_tests_dict.items():
            if self.matches_test(testname, disabled_test):
                platform_to_conditional: Dict = {
                    "mac": IS_MACOS,
                    "macos": IS_MACOS,
                    "win": IS_WINDOWS,
                    "windows": IS_WINDOWS,
                    "linux": IS_LINUX,
                    "rocm": TEST_WITH_ROCM,  # noqa: F821
                    "xpu": TEST_XPU,  # noqa: F821
                    "asan": TEST_WITH_ASAN,  # noqa: F821
                    "dynamo": TEST_WITH_TORCHDYNAMO,  # noqa: F821
                    "inductor": TEST_WITH_TORCHINDUCTOR,  # noqa: F821
                    "slow": TEST_WITH_SLOW,  # noqa: F821
                }

                invalid_platforms = list(
                    filter(lambda p: p not in platform_to_conditional, platforms)
                )
                if len(invalid_platforms) > 0:
                    invalid_plats_str = ", ".join(invalid_platforms)
                    valid_plats = ", ".join(platform_to_conditional.keys())

                    print(
                        f"Test {disabled_test} is disabled for some unrecognized ",
                        f"platforms: [{invalid_plats_str}]. Please edit issue {issue_url} to fix the platforms ",
                        'assigned to this flaky test, changing "Platforms: ..." to a comma separated ',
                        f"subset of the following (or leave it blank to match all platforms): {valid_plats}",
                    )

                    # Sanitize the platforms list so that we continue to disable the test for any valid platforms given
                    platforms = list(
                        filter(lambda p: p in platform_to_conditional, platforms)
                    )

                if platforms == [] or any(
                    platform_to_conditional[platform] for platform in platforms
                ):
                    return (
                        f"Test is disabled because an issue exists disabling it: {issue_url}"
                        f" for {'all' if platforms == [] else ''}platform(s) {', '.join(platforms)}. "
                        "If you're seeing this on your local machine and would like to enable this test, "
                        "please make sure CI is not set and you are not using the flag --import-disabled-tests."
                    )
        return

    def pytest_collection_modifyitems(self, config, items) -> Optional[str]:
        for item in items:
            testname = item.nodeid
            reason = self.get_disable_reason(testname)
            if reason is not None:
                skip = pytest.mark.skip(reason=reason)
                item.add_marker(skip)
