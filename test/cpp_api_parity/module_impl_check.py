# The purpose of this test is to check that we have implementation parity between
# a Python `torch.nn` module and its corresponding C++ `torch::nn` module. Concretely,
# this test does the following:
#
# 1. Get a test params dict from common_nn.py, run forward and backward on the
# Python module created using the test params.
#
# 2. Serialize the Python module's parameters / buffers and its forward input
# arguments, deserialize them in C++ and load them into the C++ module.
#
# 3. Run the same forward and backward passes on the C++ module, and serialize
# the C++ module's forward output and backward gradients.
#
# 4. Compare Python/C++ module's forward output and backward gradients. If they
# are the same, then we have implementation parity between Python/C++ module.

import tempfile
import pprint

import torch
from cpp_api_parity.utils import TorchNNModuleTestParams, TORCH_NN_COMMON_TEST_HARNESS, \
    compile_cpp_code_inline, add_test, compute_cpp_args_construction_stmts_and_forward_arg_symbols, \
    compute_arg_dict, decorate_test_fn, is_torch_nn_functional_test, try_remove_folder
from cpp_api_parity.sample_module import SAMPLE_MODULE_CPP_SOURCE
from cpp_api_parity.module_utils import test_forward_backward, TORCH_NN_MODULE_TEST_FORWARD_BACKWARD

def compute_module_name(test_params_dict):
    fullname = test_params_dict.get('fullname', None)
    if fullname:
        module_name = fullname.split('_')[0]
    else:
        module_name = test_params_dict.get('module_name')
    return module_name

def process_test_params_for_module(test_params_dict, device, test_instance_class):
    module_name = compute_module_name(test_params_dict)
    test_params_dict['constructor'] = test_params_dict.get('constructor', getattr(torch.nn, module_name))
    test_instance = test_instance_class(**test_params_dict)
    assert test_instance.get_name().startswith('test_')
    # Example output: `BCELoss_weights_cuda`
    module_variant_name = test_instance.get_name()[5:] + (('_' + device) if device != 'cpu' else '')

    if 'constructor_args' in test_params_dict:
        assert 'cpp_constructor_args' in test_params_dict, (
            "If `constructor_args` is present in test params dict, to enable C++ API parity test, "
            "`cpp_constructor_args` must be present in:\n{}"
            "If you are interested in adding the C++ API parity test, please see:\n"
            "NOTE [How to check NN module / functional API parity between Python and C++ frontends]. \n"
            "If not, please add `test_cpp_api_parity=False` to the test params dict and file an issue about this."
        ).format(pprint.pformat(test_params_dict))

    return TorchNNModuleTestParams(
        module_name=module_name,
        module_variant_name=module_variant_name,
        test_instance=test_instance,
        cpp_constructor_args=test_params_dict.get('cpp_constructor_args', ''),
        arg_dict=compute_arg_dict(test_params_dict, test_instance),
        has_parity=test_params_dict.get('has_parity', True),
        device=device,
        cpp_tmp_folder=tempfile.mkdtemp(),
    )

def write_test_to_test_class(
        unit_test_class, test_params_dict, test_instance_class, parity_table, devices):
    assert not is_torch_nn_functional_test(test_params_dict)

    module_name = compute_module_name(test_params_dict)

    assert hasattr(torch.nn, module_name), (
        "`torch.nn` doesn't have module `{}`. "
        "If you are adding a new test, please set `fullname` using format `ModuleName_desc` "
        "or set `module_name` using format `ModuleName` in the module test dict:\n{}"
    ).format(module_name, pprint.pformat(test_params_dict))

    module_full_name = 'torch::nn::' + module_name

    assert module_full_name in parity_table['torch::nn'], (
        "Please add `{}` entry to `torch::nn` section of `test/cpp_api_parity/parity-tracker.md`. "
        "(Discovered while processing\n{}.)").format(module_full_name, pprint.pformat(test_params_dict))

    for device in devices:
        test_params = process_test_params_for_module(
            test_params_dict=test_params_dict,
            device=device,
            test_instance_class=test_instance_class,
        )
        try_remove_folder(test_params.cpp_tmp_folder)
        unit_test_name = 'test_torch_nn_{}'.format(test_params.module_variant_name)
        unit_test_class.module_test_params_map[unit_test_name] = test_params

        def test_fn(self):
            test_forward_backward(
                unit_test_class=self, test_params=unit_test_class.module_test_params_map[self._testMethodName],
                cpp_module=unit_test_class.module_impl_check_cpp_module)

        test_fn = decorate_test_fn(
            test_fn=test_fn,
            test_cuda=test_params_dict.get('test_cuda', True),
            has_impl_parity=parity_table['torch::nn'][module_full_name][0] and
            test_params_dict.get('has_parity', True),
            device=device)

        add_test(unit_test_class, unit_test_name, test_fn)

def generate_test_cpp_sources(test_params, template):
    device = test_params.device

    cpp_constructor_args = test_params.cpp_constructor_args
    if cpp_constructor_args != '':
        cpp_constructor_args = '({})'.format(cpp_constructor_args)

    cpp_args_construction_stmts, cpp_forward_args_symbols = \
        compute_cpp_args_construction_stmts_and_forward_arg_symbols(test_params)

    test_cpp_sources = template.substitute(
        module_variant_name=test_params.module_variant_name,
        module_qualified_name='torch::nn::{}'.format(test_params.module_name),
        cpp_args_construction_stmts=";\n  ".join(cpp_args_construction_stmts),
        cpp_constructor_args=cpp_constructor_args,
        cpp_forward_args_symbols=", ".join(cpp_forward_args_symbols),
        device=device,
    )
    return test_cpp_sources

# Build all C++ tests together, instead of once per test.
def build_cpp_tests(unit_test_class, print_cpp_source=False):
    assert len(unit_test_class.module_test_params_map) > 0
    cpp_sources = TORCH_NN_COMMON_TEST_HARNESS + SAMPLE_MODULE_CPP_SOURCE
    functions = []
    for test_name, test_params in unit_test_class.module_test_params_map.items():
        cpp_sources += generate_test_cpp_sources(
            test_params=test_params, template=TORCH_NN_MODULE_TEST_FORWARD_BACKWARD)
        functions.append('{}_test_forward_backward'.format(test_params.module_variant_name))
    if print_cpp_source:
        print(cpp_sources)

    cpp_module = compile_cpp_code_inline(
        name='module_impl_check',
        cpp_sources=cpp_sources,
        functions=functions)
    unit_test_class.module_impl_check_cpp_module = cpp_module
