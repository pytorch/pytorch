import json

import pytest
from _pytest.config.argparsing import Parser


def pytest_addoptions(parser: Parser):
    """Add options to control sharding."""
    group = parser.getgroup("shard")
    group.addoption(
        "--disabled-tests-file",
        type=str,
    )
    group.addoption(
        "--rerun-disabled-tests",
        action="store_true",
    )


class DisabledTestsPlugin:
    def __init__(self, config):
        self.config = config
        self.file = config.getoption("disabled_tests_file")
        self.rerun_disabled_tests = config.getoption("rerun_disabled_tests")

    def pytest_collection_modifyitems(self, config, items):
        try:
            from torch.testing._internal.common_utils import (
                check_if_disabled,
                IS_SANDCASTLE,
            )
        except ImportError as e:
            print(
                "Used --disabled-tests-file but failed to import torch.testing._internal.common_utils, aborting"
            )
            raise e
        if IS_SANDCASTLE:
            return

        with open(self.file) as f:
            disabled_tests = json.load(f)

        for item in items:
            testname = item.name
            classname = item.nodeid.split("::")[-2]
            is_disabled, skip_msg = check_if_disabled(
                classname, testname, disabled_tests
            )
            if is_disabled and not self.rerun_disabled_tests:
                # Skip the disabled test when not running under --rerun-disabled-tests verification mode
                item.add_marker(pytest.mark.skip(reason=skip_msg))

            if not is_disabled and self.rerun_disabled_tests:
                skip_msg = (
                    "Test is enabled but --rerun-disabled-tests verification mode is set, so only"
                    " disabled tests are run"
                )
                item.add_marker(pytest.mark.skip(reason=skip_msg))
