# The purpose of this test is to check that we have implementation parity between
# a Python `torch.nn.functional` function and its corresponding C++ `torch::nn::functional`
# function. Concretely, this test does the following:
#
# 1. Get a test params dict from common_nn.py, run forward pass on the Python functional
# created using the test params.
#
# 2. Serialize the Python functional's forward input arguments, deserialize them
# in C++ and use them as input for the C++ functional's forward pass.
#
# 3. Run the forward pass on the C++ functional, and serialize the C++ functional's
# forward output.
#
# 4. Compare Python/C++ functional's forward output. If they are the same, then we
# have implementation parity between Python/C++ module.

import tempfile
import shutil
from string import Template
import unittest
import re

import torch
import torch.testing._internal.common_nn as common_nn
from torch.testing._internal.common_cuda import TEST_CUDA
from cpp_api_parity.utils import TorchNNFunctionalTestParams, CppArg, TORCH_NN_COMMON_TEST_HARNESS, \
    compile_cpp_code_inline, convert_to_list, set_python_tensors_requires_grad, move_python_tensors_to_device, \
    has_test, add_test, set_cpp_tensors_requires_grad, move_cpp_tensors_to_device, is_criterion_test, \
    compute_cpp_args_construction_stmts_and_forward_arg_symbols, serialize_arg_dict_as_script_module, \
    compute_arg_dict, decorate_test_fn, compute_temp_file_path, generate_error_msg
from cpp_api_parity import torch_nn_functionals

# NN tests use double as the default dtype
torch.set_default_dtype(torch.double)

# Expected substitutions:
#
# ${functional_variant_name}
# ${cpp_tmp_folder}
# ${cpp_args_construction_stmts}
# ${cpp_function_call}
TORCH_NN_FUNCTIONAL_TEST_FORWARD = Template("""
void ${functional_variant_name}_test_forward(
    const std::string& arg_dict_file_path,
    const std::string& forward_output_file_path) {
  pybind11::gil_scoped_release no_gil;

  namespace F = torch::nn::functional;

  // Declare arguments
  auto arg_dict = load_dict_from_file(arg_dict_file_path);
  ${cpp_args_construction_stmts};

  // Some functionals (such as `F::rrelu`) create random tensors in their call path.
  // To make sure the random tensors created are the same in Python/C++, we need
  // to set the RNG seed manually.
  torch::manual_seed(0);

  // Run function with arguments
  auto cpp_output = ${cpp_function_call};

  // Save the output into a file to be compared in Python later
  write_ivalue_to_file(torch::IValue(cpp_output), forward_output_file_path);
}
""")

def run_forward(unit_test_class, test_params):
    device = test_params.device

    inputs = set_python_tensors_requires_grad([arg_value for _, arg_value in test_params.arg_dict['input']])
    inputs = inputs + [arg_value for _, arg_value in test_params.arg_dict['target']]
    inputs = inputs + [arg_value for _, arg_value in test_params.arg_dict['extra_args']]
    inputs = move_python_tensors_to_device(inputs, device)

    # Some functionals (such as `F.rrelu`) create random tensors in their call path.
    # To make sure the random tensors created are the same in Python/C++, we need
    # to set the RNG seed manually.
    torch.manual_seed(0)
    python_output = test_params.test_instance.constructor()(*inputs)

    return python_output

def test_forward(unit_test_class, test_params):
    functional_variant_name = test_params.functional_variant_name
    cpp_tmp_folder = test_params.cpp_tmp_folder

    # Run forward on Python functional
    python_output = run_forward(unit_test_class, test_params)

    # Save Python arguments to be used from C++ function
    arg_dict_file_path = compute_temp_file_path(cpp_tmp_folder, functional_variant_name, 'arg_dict')
    serialize_arg_dict_as_script_module(test_params.arg_dict).save(arg_dict_file_path)

    cpp_test_name = '{}_{}'.format(test_params.functional_variant_name, 'test_forward')
    cpp_test_fn = getattr(unit_test_class.functional_impl_check_cpp_module, cpp_test_name)

    def run_cpp_test_fn_and_check_output():
        forward_output_file_path = compute_temp_file_path(cpp_tmp_folder, functional_variant_name, 'forward_output')

        cpp_test_fn(arg_dict_file_path, forward_output_file_path)
        cpp_output = torch.load(forward_output_file_path)

        # Check that forward outputs are equal
        unit_test_class.assertTrue(
            torch.allclose(python_output, cpp_output),
            generate_error_msg("forward output", cpp_output, python_output))

    run_cpp_test_fn_and_check_output()

    # Remove temporary folder that stores C++ outputs
    shutil.rmtree(cpp_tmp_folder)

def compute_functional_name(test_params_dict):
    def camel_case_to_snake_case(camel_case_str):
        return re.sub(r'(?<!^)(?=[A-Z])', '_', camel_case_str).lower()

    if 'cpp_options_args' in test_params_dict:
        # Expected format: `F::FunctionalFuncOptions(...)`
        return camel_case_to_snake_case(test_params_dict['cpp_options_args'].split('(')[0].replace('F::', '').replace('FuncOptions', ''))
    elif 'cpp_function_call' in test_params_dict:
        # Expected format: `F::functional_name(...)`
        return test_params_dict['cpp_function_call'].split('(')[0].replace('F::', '')
    else:
        raise RuntimeError(
            "`cpp_options_args` or `cpp_function_call` entry must be present in test params dict: {}".format(
                test_params_dict))

def compute_cpp_function_call(test_params_dict, arg_dict, functional_name):
    if 'cpp_function_call' in test_params_dict:
        return test_params_dict['cpp_function_call']
    elif 'cpp_options_args' in test_params_dict:
        cpp_forward_args_symbols = [arg_name for arg_name, _ in arg_dict['input'] + arg_dict['target'] + arg_dict['extra_args']]
        return 'F::{}({}, {})'.format(functional_name, ", ".join(cpp_forward_args_symbols), test_params_dict['cpp_options_args'])
    else:
        raise RuntimeError(
            "`cpp_options_args` or `cpp_function_call` entry must be present in test params dict: {}".format(
                test_params_dict))

def process_test_params_for_functional(test_params_dict, device, test_instance_class):
    test_instance = test_instance_class(**test_params_dict)
    functional_name = compute_functional_name(test_params_dict)
    assert test_instance.get_name().startswith('test_')
    functional_variant_name = test_instance.get_name()[5:] + (('_' + device) if device != 'cpu' else '')
    arg_dict = compute_arg_dict(test_params_dict, test_instance)

    return TorchNNFunctionalTestParams(
        functional_name=functional_name,
        functional_variant_name=functional_variant_name,
        test_instance=test_instance,
        cpp_function_call=compute_cpp_function_call(test_params_dict, arg_dict, functional_name),
        arg_dict=arg_dict,
        has_parity=test_params_dict.get('has_parity', True),
        device=device,
        cpp_tmp_folder=tempfile.mkdtemp(),
    )

torch_nn_test_params_map = {}

def test_torch_nn_functional_variant(unit_test_class, test_params):
    test_forward(unit_test_class, test_params)

def add_torch_nn_functional_impl_parity_tests(parity_table, unit_test_class, test_params_dicts, test_instance_class, devices):
    for test_params_dict in test_params_dicts:
        # Skip all `torch.nn` module tests, since they are handled by another test suite.
        if not 'FunctionalModule' in str(test_params_dict.get('constructor', '')):
            continue

        functional_name = compute_functional_name(test_params_dict)

        assert hasattr(torch.nn.functional, functional_name), \
            "`torch.nn.functional` doesn't have function `{}`. (Discovered while processing {}.)".format(
                functional_name, test_params_dict)

        functional_full_name = 'F::' + functional_name

        assert functional_full_name in parity_table['torch::nn::functional'], \
            "Please add `{}` entry to `torch::nn::functional` section of `test/cpp_api_parity/parity-tracker.md`. (Discovered while processing {}.)".format(
                functional_full_name, test_params_dict)

        for device in devices:
            test_params = process_test_params_for_functional(
                test_params_dict=test_params_dict,
                device=device,
                test_instance_class=test_instance_class,
            )
            test_name = 'test_torch_nn_functional_{}'.format(test_params.functional_variant_name)
            torch_nn_test_params_map[test_name] = test_params

            def test_fn(self):
                test_torch_nn_functional_variant(unit_test_class=self, test_params=torch_nn_test_params_map[self._testMethodName])

            test_fn = decorate_test_fn(
                test_fn=test_fn,
                test_cpp_api_parity=test_params_dict.get('test_cpp_api_parity', True),
                test_cuda=test_params_dict.get('test_cuda', True),
                has_impl_parity=parity_table['torch::nn::functional'][functional_full_name][0] and test_params_dict.get('has_parity', True),
                device=device)

            add_test(unit_test_class, test_name, test_fn)

def add_tests(unit_test_class, test_params_dicts, test_instance_class, parity_table, devices):
    add_torch_nn_functional_impl_parity_tests(
        parity_table=parity_table,
        unit_test_class=unit_test_class,
        test_params_dicts=test_params_dicts,
        test_instance_class=test_instance_class,
        devices=devices)

def generate_test_cpp_sources(test_params, template):
    cpp_args_construction_stmts, _ = compute_cpp_args_construction_stmts_and_forward_arg_symbols(test_params)

    test_cpp_sources = template.substitute(
        functional_variant_name=test_params.functional_variant_name,
        cpp_args_construction_stmts=";\n  ".join(cpp_args_construction_stmts),
        cpp_function_call=test_params.cpp_function_call,
        cpp_tmp_folder=test_params.cpp_tmp_folder,
    )
    return test_cpp_sources

# Build all C++ tests together, instead of once per test.
def build_cpp_tests(unit_test_class, print_cpp_source=False):
    if len(torch_nn_test_params_map) > 0:
        cpp_sources = TORCH_NN_COMMON_TEST_HARNESS
        functions = []
        functionals_added_metadata_cpp_sources = set()
        for test_name, test_params in torch_nn_test_params_map.items():
            if not test_params.functional_name in functionals_added_metadata_cpp_sources:
                cpp_sources += torch_nn_functionals.functional_metadata_map.get(test_params.functional_name, torch_nn_functionals.TorchNNFunctionalMetadata()).cpp_sources
                functionals_added_metadata_cpp_sources.add(test_params.functional_name)
            cpp_sources += generate_test_cpp_sources(test_params=test_params, template=TORCH_NN_FUNCTIONAL_TEST_FORWARD)
            functions.append('{}_{}'.format(test_params.functional_variant_name, 'test_forward'))
        if print_cpp_source:
            print(cpp_sources)

        cpp_module = compile_cpp_code_inline(
            name='functional_impl_check',
            cpp_sources=cpp_sources,
            functions=functions)
        unit_test_class.functional_impl_check_cpp_module = cpp_module
