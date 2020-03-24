MESSAGE_HOW_TO_FIX_CPP_PARITY_TEST_FAILURE = '''
What should I do when C++ API parity test is failing?

- If you are changing the implementation of an existing `torch.nn` module / `torch.nn.functional` function:
Answer: Ideally you should also change the C++ API implementation for that module / function
(you can start by searching for the module / function name in `torch/csrc/api/` folder).

- If you are adding a new test for an existing `torch.nn` module / `torch.nn.functional` function:
Answer: Ideally you should fix the C++ API implementation for that module / function
to exactly match the Python API implementation (you can start by searching for the module /
function name in `torch/csrc/api/` folder).

- If you are adding a test for a *new* `torch.nn` module / `torch.nn.functional` function:
Answer: Ideally you should add the corresponding C++ API implementation for that module / function,
and it should exactly match the Python API implementation. (We have done a large effort on this
which is tracked at https://github.com/pytorch/pytorch/issues/25883.)

However, if any of the above is proven to be too complicated, you can just add
`test_cpp_api_parity=False` to any failing test in `torch/testing/_internal/common_nn.py`,
and the C++ API parity test will be skipped accordingly. Note that you should
also file an issue when you do this.
'''
import cpp_api_parity.utils
cpp_api_parity.utils.extra_msg_on_failure = MESSAGE_HOW_TO_FIX_CPP_PARITY_TEST_FAILURE

import os

import torch.testing._internal.common_utils as common
import torch.testing._internal.common_nn as common_nn
from cpp_api_parity.parity_table_parser import parse_parity_tracker_table
from cpp_api_parity import module_impl_check, functional_impl_check, sample_module, sample_functional

# NOTE: turn this on if you want to print source code of all C++ tests (e.g. for debugging purpose)
print_cpp_source = False

devices = ['cpu', 'cuda']

PARITY_TABLE_PATH = os.path.join(os.path.dirname(__file__), 'cpp_api_parity', 'parity-tracker.md')

parity_table = parse_parity_tracker_table(PARITY_TABLE_PATH)

class TestCppApiParity(common.TestCase):
    pass

for test_params_dicts, test_instance_class in [
    (sample_module.module_tests, common_nn.ModuleTest),
    (sample_functional.functional_tests, common_nn.NewModuleTest),
    (common_nn.module_tests, common_nn.ModuleTest),
    (common_nn.new_module_tests, common_nn.NewModuleTest),
    (common_nn.criterion_tests, common_nn.CriterionTest),
    (common_nn.new_criterion_tests, common_nn.NewCriterionTest),
]:
    module_impl_check.add_tests(TestCppApiParity, test_params_dicts, test_instance_class, parity_table, devices)
    functional_impl_check.add_tests(TestCppApiParity, test_params_dicts, test_instance_class, parity_table, devices)

# Assert that there exists auto-generated tests for `SampleModule` and `sample_functional`.
assert len([name for name in TestCppApiParity.__dict__ if 'SampleModule' in name]) == \
    len(sample_module.module_tests) * len(devices)
assert len([name for name in TestCppApiParity.__dict__ if 'sample_functional' in name]) == \
    len(sample_functional.functional_tests) * len(devices)

if __name__ == "__main__":
    module_impl_check.build_cpp_tests(TestCppApiParity, print_cpp_source=print_cpp_source)
    functional_impl_check.build_cpp_tests(TestCppApiParity, print_cpp_source=print_cpp_source)
    common.run_tests()
