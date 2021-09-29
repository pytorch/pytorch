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
import types
import os

import torch
import torch.testing._internal.common_modules as common_modules
from cpp_api_parity.utils import TorchNNModuleTestParams, TORCH_NN_COMMON_TEST_HARNESS, \
    compile_cpp_code_inline, set_python_tensors_requires_grad, move_python_tensors_to_device, \
    add_test, compute_cpp_args_construction_stmts_and_forward_arg_symbols, serialize_arg_dict_as_script_module, \
    decorate_test_fn, compute_temp_file_path, generate_error_msg, \
    try_remove_folder, CppArg, convert_to_list
from cpp_api_parity.sample_module import SAMPLE_MODULE_CPP_SOURCE
from cpp_api_parity.module_impl_check import TORCH_NN_MODULE_TEST_FORWARD_BACKWARD

def run_python_forward_backward(unit_test_class, test_params):
    device = test_params.device
    module_info, sample = test_params.test_instance
    args, kwargs = sample.constructor_input.args, sample.constructor_input.kwargs
    module = module_info.module_cls(*args, **kwargs).to(device)

    inputs = set_python_tensors_requires_grad(move_python_tensors_to_device(
        [arg_value for _, arg_value in test_params.arg_dict['input']], device))
    inputs += move_python_tensors_to_device(
        [arg_value for _, arg_value in test_params.arg_dict['target']], device)
    inputs += move_python_tensors_to_device(
        [arg_value for _, arg_value in test_params.arg_dict['extra_args']], device)

    # Some modules (such as `RReLU`) create random tensors in their forward pass.
    # To make sure the random tensors created are the same in Python/C++, we need
    # to set the RNG seed manually.
    torch.manual_seed(0)

    # Forward pass
    python_output = module(*inputs)

    # NOTE: This is a workaround to allow any module to be traced.
    # We can do this because we are only interested in transferring
    # the Python module's parameters and buffers to the C++ module.
    module.forward = types.MethodType(lambda self, input: input, module)
    script_module = torch.jit.trace(module, torch.tensor(0))

    # Backward pass
    python_output.sum().backward()

    # Put all gradients into a dict, to be compared later
    python_grad_dict = {}
    for name, param in module.named_parameters():
        grad = param.grad
        if grad.is_sparse:
            python_grad_dict[name + "_grad_indices"] = grad.coalesce().indices()
            python_grad_dict[name + "_grad_values"] = grad.coalesce().values()
        else:
            python_grad_dict[name + "_grad"] = grad

    return script_module, python_output, python_grad_dict

def test_forward_backward(unit_test_class, test_params):
    module_variant_name = test_params.module_variant_name
    cpp_tmp_folder = test_params.cpp_tmp_folder
    # Remove the temporary folder if it exists already
    try_remove_folder(cpp_tmp_folder)
    os.mkdir(cpp_tmp_folder)

    # Run forward and backward on Python module
    script_module, python_output, python_grad_dict = run_python_forward_backward(unit_test_class, test_params)

    # Save Python module and arguments to be used from C++ function
    module_file_path = compute_temp_file_path(cpp_tmp_folder, module_variant_name, 'module')
    arg_dict_file_path = compute_temp_file_path(cpp_tmp_folder, module_variant_name, 'arg_dict')
    script_module.save(module_file_path)
    serialize_arg_dict_as_script_module(test_params.arg_dict).save(arg_dict_file_path)

    cpp_test_name = '{}_test_forward_backward'.format(test_params.module_variant_name)
    cpp_test_fn = getattr(unit_test_class.module_info_impl_check_cpp_module, cpp_test_name)

    def run_cpp_test_fn_and_check_output():
        forward_output_file_path = compute_temp_file_path(cpp_tmp_folder, module_variant_name, 'forward_output')
        backward_grad_dict_file_path = compute_temp_file_path(cpp_tmp_folder, module_variant_name, 'backward_grad_dict')

        cpp_test_fn(arg_dict_file_path, module_file_path, forward_output_file_path, backward_grad_dict_file_path)
        cpp_output = torch.load(forward_output_file_path)
        cpp_grad_dict = torch.load(backward_grad_dict_file_path)

        # Check that forward outputs are equal
        unit_test_class.assertEqual(python_output, cpp_output,
                                    msg=generate_error_msg("forward output", cpp_output, python_output))

        # Check that module parameter gradients are equal after backward pass
        unit_test_class.assertEqual(
            len(python_grad_dict), len(cpp_grad_dict),
            msg=generate_error_msg("# of parameters", len(cpp_grad_dict), len(python_grad_dict)))
        for key in python_grad_dict:
            param_name = None
            for suffix in ['_grad', '_grad_indices', '_grad_values']:
                if key.endswith(suffix):
                    param_name = key[:-len(suffix)]
                    break
            assert param_name is not None
            sparsity_str = 'sparse' if key.endswith('_grad_indices') or key.endswith('_grad_values') else 'dense'

            unit_test_class.assertTrue(
                key in cpp_grad_dict,
                msg=generate_error_msg(
                    "\"Does module have a parameter named `{}` with {} gradient?\"".format(param_name, sparsity_str),
                    False, True))
            unit_test_class.assertEqual(
                python_grad_dict[key], cpp_grad_dict[key],
                msg=generate_error_msg(
                    "`{}`'s {} gradient (`{}`)".format(param_name, sparsity_str, key),
                    cpp_grad_dict[key], python_grad_dict[key]))

    run_cpp_test_fn_and_check_output()

    # Remove temporary folder that stores C++ outputs
    try_remove_folder(cpp_tmp_folder)

def process_test_params_for_module(module_info, sample, device):
    module_name = module_info.name.replace('nn_', '')
    module_variant_name = module_name + sample.desc + (('_' + device) if device != 'cpu' else '')

    def compute_arg_dict(sample):
        arg_dict = {
            'input': [],
            'target': [],
            'extra_args': [],
            'other': [],
        }

        def put_args_into_arg_dict(arg_type, arg_type_prefix, args):
            for i, arg in enumerate(args):
                arg_dict[arg_type].append(CppArg(name=arg_type_prefix + str(i), value=arg))

        put_args_into_arg_dict('input', 'i', convert_to_list(sample.forward_input.args) +
                               convert_to_list(sample.forward_input.kwargs))
        return arg_dict

    arg_dict = compute_arg_dict(sample)

    return TorchNNModuleTestParams(
        module_name=module_name,
        module_variant_name=module_variant_name,
        test_instance=(module_info, sample),
        cpp_constructor_args=getattr(sample, 'cpp_constructor_args', ''),
        arg_dict=arg_dict,
        has_parity=True,
        device=device,
        cpp_tmp_folder=tempfile.mkdtemp(),
    )


def write_test_to_test_class(
        unit_test_class, module_info, devices):
    assert isinstance(module_info, common_modules.ModuleInfo)
    module_name = module_info.name.replace('nn_', '')
    assert hasattr(torch.nn, module_name), (
        "`torch.nn` doesn't have module `{}` for module_info"
    ).format(module_name)

    for device in devices:
        for sample in module_info.module_inputs_func(module_info, device='cpu',
                                                     dtype=torch.get_default_dtype(), requires_grad=False):
            if sample.test_cpp_parity:
                test_params = process_test_params_for_module(
                    module_info=module_info,
                    sample=sample,
                    device=device,
                )
                try_remove_folder(test_params.cpp_tmp_folder)
                unit_test_name = 'test_torch_module_info_nn_{}'.format(test_params.module_variant_name)
                unit_test_class.module_info_test_params_map[unit_test_name] = test_params

                def test_fn(self):
                    test_forward_backward(
                        unit_test_class=self, test_params=unit_test_class.module_info_test_params_map[self._testMethodName])

                test_fn = decorate_test_fn(
                    test_fn=test_fn,
                    test_cuda=True,
                    has_impl_parity=True,
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
    assert len(unit_test_class.module_info_test_params_map) > 0
    cpp_sources = TORCH_NN_COMMON_TEST_HARNESS + SAMPLE_MODULE_CPP_SOURCE
    functions = []
    for test_name, test_params in unit_test_class.module_info_test_params_map.items():
        cpp_sources += generate_test_cpp_sources(
            test_params=test_params, template=TORCH_NN_MODULE_TEST_FORWARD_BACKWARD)
        functions.append('{}_test_forward_backward'.format(test_params.module_variant_name))
    if print_cpp_source:
        print(cpp_sources)

    cpp_module = compile_cpp_code_inline(
        name='module_info_impl_check',
        cpp_sources=cpp_sources,
        functions=functions)
    unit_test_class.module_info_impl_check_cpp_module = cpp_module
