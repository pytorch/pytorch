import os
from string import Template
import types

import torch

from cpp_api_parity.utils import TorchNNModuleInfoTestParams, TorchNNModuleTestParams, \
    set_python_tensors_requires_grad, move_python_tensors_to_device, \
    serialize_arg_dict_as_script_module, \
    compute_temp_file_path, generate_error_msg, \
    try_remove_folder, compute_cpp_args_construction_stmts_and_forward_arg_symbols

# Expected substitutions:
#
# ${module_variant_name}  (e.g. `Linear_no_bias_cpu`)
# ${module_qualified_name}  (e.g. `torch::nn::Linear`)
# ${cpp_args_construction_stmts}
# ${cpp_constructor_args}
# ${device}
# ${cpp_forward_args_symbols}
TORCH_NN_MODULE_TEST_FORWARD_BACKWARD = Template("""
void ${module_variant_name}_test_forward_backward(
    const std::string& arg_dict_file_path,
    const std::string& module_file_path,
    const std::string& forward_output_file_path,
    const std::string& backward_grad_dict_file_path) {
  pybind11::gil_scoped_release no_gil;

  // Declare arguments
  auto arg_dict = load_dict_from_file(arg_dict_file_path);
  ${cpp_args_construction_stmts};

  // Construct module and load params/buffers from Python module
  ${module_qualified_name} module${cpp_constructor_args};
  module->to(std::string("${device}"));
  torch::load(module, module_file_path);

  // Some modules (such as `RReLU`) create random tensors in their forward pass.
  // To make sure the random tensors created are the same in Python/C++, we need
  // to set the RNG seed manually.
  torch::manual_seed(0);

  // Forward pass
  auto cpp_output = module(${cpp_forward_args_symbols});

  // Save the output into a file to be compared in Python later
  write_ivalue_to_file(torch::IValue(cpp_output), forward_output_file_path);

  // Backward pass
  cpp_output.sum().backward();

  // Put all gradients into a c10::Dict, save it into a file to be compared in Python later
  c10::Dict<std::string, torch::Tensor> grad_dict;
  for (const auto& param : module->named_parameters()) {
    torch::Tensor grad = param.value().grad();
    if (grad.is_sparse()) {
      grad_dict.insert(param.key() + "_grad_indices", grad.coalesce().indices());
      grad_dict.insert(param.key() + "_grad_values", grad.coalesce().values());
    } else {
      grad_dict.insert(param.key() + "_grad", grad);
    }
  }

  write_ivalue_to_file(torch::IValue(grad_dict), backward_grad_dict_file_path);
}
""")


def run_python_forward_backward(unit_test_class, test_params):
    device = test_params.device
    # Construct Module
    if isinstance(test_params, TorchNNModuleInfoTestParams):
        module_info = test_params.module_info
        sample = test_params.sample
        args, kwargs = sample.constructor_input.args, sample.constructor_input.kwargs
        module = module_info.module_cls(*args, **kwargs).to(device)
    elif isinstance(test_params, TorchNNModuleTestParams):
        constructor_args = test_params.test_instance.constructor_args
        module = test_params.test_instance.constructor(*constructor_args).to(device)
    else:
        raise TypeError("Invalid type for test_params, should be either TorchNNModuleInfoTestParams or TorchNNModuleTestParams,"
                        " but received ", type(test_params))

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


def test_forward_backward(unit_test_class, test_params, cpp_module):
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
    cpp_test_fn = getattr(cpp_module, cpp_test_name)

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
