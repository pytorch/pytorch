# Owner(s): ["module: cpp"]

import torch
# NN tests use double as the default dtype
torch.set_default_dtype(torch.double)

import os

import torch.testing._internal.common_utils as common
import torch.testing._internal.common_nn as common_nn
from cpp_api_parity.parity_table_parser import parse_parity_tracker_table
from cpp_api_parity.utils import is_torch_nn_functional_test
from cpp_api_parity import module_impl_check, functional_impl_check, sample_module, sample_functional

# NOTE: turn this on if you want to print source code of all C++ tests (e.g. for debugging purpose)
PRINT_CPP_SOURCE = False

devices = ['cpu', 'cuda']

PARITY_TABLE_PATH = os.path.join(os.path.dirname(__file__), 'cpp_api_parity', 'parity-tracker.md')

parity_table = parse_parity_tracker_table(PARITY_TABLE_PATH)

class TestCppApiParity(common.TestCase):
    module_test_params_map = {}
    functional_test_params_map = {}

expected_test_params_dicts = []

if not common.IS_ARM64:
    for test_params_dicts, test_instance_class in [
        (sample_module.module_tests, common_nn.NewModuleTest),
        (sample_functional.functional_tests, common_nn.NewModuleTest),
        (common_nn.module_tests, common_nn.NewModuleTest),
        (common_nn.new_module_tests, common_nn.NewModuleTest),
        (common_nn.criterion_tests, common_nn.CriterionTest),
    ]:
        for test_params_dict in test_params_dicts:
            if test_params_dict.get('test_cpp_api_parity', True):
                if is_torch_nn_functional_test(test_params_dict):
                    functional_impl_check.write_test_to_test_class(
                        TestCppApiParity, test_params_dict, test_instance_class, parity_table, devices)
                else:
                    module_impl_check.write_test_to_test_class(
                        TestCppApiParity, test_params_dict, test_instance_class, parity_table, devices)
                expected_test_params_dicts.append(test_params_dict)

    # Assert that all NN module/functional test dicts appear in the parity test
    assert len([name for name in TestCppApiParity.__dict__ if 'test_torch_nn_' in name]) == \
        len(expected_test_params_dicts) * len(devices)

    # Assert that there exists auto-generated tests for `SampleModule` and `sample_functional`.
    # 4 == 2 (number of test dicts that are not skipped) * 2 (number of devices)
    assert len([name for name in TestCppApiParity.__dict__ if 'SampleModule' in name]) == 4
    # 4 == 2 (number of test dicts that are not skipped) * 2 (number of devices)
    assert len([name for name in TestCppApiParity.__dict__ if 'sample_functional' in name]) == 4

    module_impl_check.build_cpp_tests(TestCppApiParity, print_cpp_source=PRINT_CPP_SOURCE)
    functional_impl_check.build_cpp_tests(TestCppApiParity, print_cpp_source=PRINT_CPP_SOURCE)

if __name__ == "__main__":
    common.run_tests()
