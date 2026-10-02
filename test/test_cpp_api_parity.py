# Owner(s): ["module: cpp"]


import os

from cpp_api_parity import (
    functional_impl_check,
    module_impl_check,
    sample_functional,
    sample_module,
)
from cpp_api_parity.parity_table_parser import parse_parity_tracker_table
from cpp_api_parity.utils import is_torch_nn_functional_test

import torch
import torch.testing._internal.common_nn as common_nn
import torch.testing._internal.common_utils as common
from torch.testing._internal.common_device_type import instantiate_device_type_tests


# NOTE: turn this on if you want to print source code of all C++ tests (e.g. for debugging purpose)
PRINT_CPP_SOURCE = False

PARITY_TABLE_PATH = os.path.join(
    os.path.dirname(__file__), "cpp_api_parity", "parity-tracker.md"
)

parity_table = parse_parity_tracker_table(PARITY_TABLE_PATH)


@torch.testing._internal.common_utils.markDynamoStrictTest
class TestCppApiParity(common.TestCase):
    module_test_params_map = {}
    functional_test_params_map = {}
    module_test_param_factories = {}
    functional_test_param_factories = {}

    def test_build_cpp(self, device):
        module_impl_check.prepare_test_params(self.__class__, torch.device(device).type)
        functional_impl_check.prepare_test_params(
            self.__class__, torch.device(device).type
        )
        module_impl_check.build_cpp_tests(
            self.__class__, print_cpp_source=PRINT_CPP_SOURCE
        )
        functional_impl_check.build_cpp_tests(
            self.__class__, print_cpp_source=PRINT_CPP_SOURCE
        )


expected_test_params_dicts = []

for test_params_dicts, test_instance_class in [
    (sample_module.module_tests, common_nn.NewModuleTest),
    (sample_functional.functional_tests, common_nn.NewModuleTest),
    (common_nn.module_tests, common_nn.NewModuleTest),
    (common_nn.get_new_module_tests(), common_nn.NewModuleTest),
    (common_nn.criterion_tests, common_nn.CriterionTest),
]:
    for test_params_dict in test_params_dicts:
        if test_params_dict.get("test_cpp_api_parity", True):
            if is_torch_nn_functional_test(test_params_dict):
                functional_impl_check.write_test_to_test_class(
                    TestCppApiParity,
                    test_params_dict,
                    test_instance_class,
                    parity_table,
                )
            else:
                module_impl_check.write_test_to_test_class(
                    TestCppApiParity,
                    test_params_dict,
                    test_instance_class,
                    parity_table,
                )
            expected_test_params_dicts.append(test_params_dict)

# Assert that all NN module/functional test dicts appear in the parity test
_test_torch_nn_count = len(
    [name for name in TestCppApiParity.__dict__ if "test_torch_nn_" in name]
)
_expected_count = len(expected_test_params_dicts)
if _test_torch_nn_count != _expected_count:
    raise AssertionError(
        f"expected {_expected_count} test_torch_nn_ tests, got {_test_torch_nn_count}"
    )

# Assert that there exists auto-generated tests for `SampleModule` and `sample_functional`.
# 2 == number of test dicts that are not skipped
_expected_sample_count = 2
_sample_module_count = len(
    [name for name in TestCppApiParity.__dict__ if "SampleModule" in name]
)
if _sample_module_count != _expected_sample_count:
    raise AssertionError(
        f"expected {_expected_sample_count} SampleModule tests, got {_sample_module_count}"
    )
_sample_functional_count = len(
    [name for name in TestCppApiParity.__dict__ if "sample_functional" in name]
)
if _sample_functional_count != _expected_sample_count:
    raise AssertionError(
        f"expected {_expected_sample_count} sample_functional tests, got {_sample_functional_count}"
    )

instantiate_device_type_tests(TestCppApiParity, globals(), allow_xpu=True)

if __name__ == "__main__":
    common.TestCase._default_dtype_check_enabled = True
    common.run_tests()
