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

import os
import pprint
import tempfile
import types
from string import Template

import torch

from cpp_api_parity.sample_module import SAMPLE_MODULE_CPP_SOURCE
from cpp_api_parity.utils import (
    add_test,
    compile_cpp_code_inline,
    compute_arg_dict,
    compute_cpp_args_construction_stmts_and_forward_arg_symbols,
    compute_temp_file_path,
    decorate_test_fn,
    generate_error_msg,
    is_torch_nn_functional_test,
    move_python_tensors_to_device,
    serialize_arg_dict_as_script_module,
    set_python_tensors_requires_grad,
    TORCH_NN_COMMON_TEST_HARNESS,
    TorchNNModuleTestParams,
    try_remove_folder,
)

# Expected substitutions:
#
# ${module_variant_name}  (e.g. `Linear_no_bias_cpu`)
# ${module_qualified_name}  (e.g. `torch::nn::Linear`)
# ${cpp_args_construction_stmts}
# ${cpp_constructor_args}
# ${device}
# ${cpp_forward_args_symbols}
TORCH_NN_MODULE_TEST_FORWARD_BACKWARD = Template(
    """
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
  if (cpp_output.is_complex()) {
    cpp_output.sum().abs().backward();
  } else {
    cpp_output.sum().backward();
  }

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
"""
)


def run_python_forward_backward(unit_test_class, test_params):
    device = test_params.device
    module = test_params.test_instance.constructor(
        *test_params.test_instance.constructor_args
    ).to(device)

    inputs = set_python_tensors_requires_grad(
        move_python_tensors_to_device(
            [arg_value for _, arg_value in test_params.arg_dict["input"]], device
        )
    )
    inputs += move_python_tensors_to_device(
        [arg_value for _, arg_value in test_params.arg_dict["target"]], device
    )
    inputs += move_python_tensors_to_device(
        [arg_value for _, arg_value in test_params.arg_dict["extra_args"]], device
    )

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
    if python_output.dtype.is_complex:
        python_output.sum().abs().backward()
    else:
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
    script_module, python_output, python_grad_dict = run_python_forward_backward(
        unit_test_class, test_params
    )

    # Save Python module and arguments to be used from C++ function
    module_file_path = compute_temp_file_path(
        cpp_tmp_folder, module_variant_name, "module"
    )
    arg_dict_file_path = compute_temp_file_path(
        cpp_tmp_folder, module_variant_name, "arg_dict"
    )
    script_module.save(module_file_path)
    serialize_arg_dict_as_script_module(test_params.arg_dict).save(arg_dict_file_path)

    cpp_test_name = f"{test_params.module_variant_name}_test_forward_backward"
    cpp_test_fn = getattr(unit_test_class.module_impl_check_cpp_module, cpp_test_name)

    def run_cpp_test_fn_and_check_output():
        forward_output_file_path = compute_temp_file_path(
            cpp_tmp_folder, module_variant_name, "forward_output"
        )
        backward_grad_dict_file_path = compute_temp_file_path(
            cpp_tmp_folder, module_variant_name, "backward_grad_dict"
        )

        cpp_test_fn(
            arg_dict_file_path,
            module_file_path,
            forward_output_file_path,
            backward_grad_dict_file_path,
        )
        cpp_output = torch.load(forward_output_file_path)
        cpp_grad_dict = torch.load(backward_grad_dict_file_path)

        # Check that forward outputs are equal
        unit_test_class.assertEqual(
            python_output,
            cpp_output,
            msg=generate_error_msg("forward output", cpp_output, python_output),
        )

        # Check that module parameter gradients are equal after backward pass
        unit_test_class.assertEqual(
            len(python_grad_dict),
            len(cpp_grad_dict),
            msg=generate_error_msg(
                "# of parameters", len(cpp_grad_dict), len(python_grad_dict)
            ),
        )
        for key in python_grad_dict:
            param_name = None
            for suffix in ["_grad", "_grad_indices", "_grad_values"]:
                if key.endswith(suffix):
                    param_name = key[: -len(suffix)]
                    break
            assert param_name is not None
            sparsity_str = (
                "sparse" if key.endswith(("_grad_indices", "_grad_values")) else "dense"
            )

            unit_test_class.assertTrue(
                key in cpp_grad_dict,
                msg=generate_error_msg(
                    f'"Does module have a parameter named `{param_name}` with {sparsity_str} gradient?"',
                    False,
                    True,
                ),
            )
            unit_test_class.assertEqual(
                python_grad_dict[key],
                cpp_grad_dict[key],
                msg=generate_error_msg(
                    f"`{param_name}`'s {sparsity_str} gradient (`{key}`)",
                    cpp_grad_dict[key],
                    python_grad_dict[key],
                ),
            )

    run_cpp_test_fn_and_check_output()

    # Remove temporary folder that stores C++ outputs
    try_remove_folder(cpp_tmp_folder)


def compute_module_name(test_params_dict):
    fullname = test_params_dict.get("fullname", None)
    if fullname:
        module_name = fullname.split("_")[0]
    else:
        module_name = test_params_dict.get("module_name")
    return module_name


def process_test_params_for_module(test_params_dict, device, test_instance_class):
    module_name = compute_module_name(test_params_dict)
    test_params_dict["constructor"] = test_params_dict.get(
        "constructor", getattr(torch.nn, module_name)
    )
    test_instance = test_instance_class(**test_params_dict)
    assert test_instance.get_name().startswith("test_")
    # Example output: `BCELoss_weights_cuda`
    module_variant_name = test_instance.get_name()[5:] + (
        ("_" + device) if device != "cpu" else ""
    )

    if "constructor_args" in test_params_dict:
        assert "cpp_constructor_args" in test_params_dict, (
            "If `constructor_args` is present in test params dict, to enable C++ API parity test, "
            f"`cpp_constructor_args` must be present in:\n{pprint.pformat(test_params_dict)}"
            "If you are interested in adding the C++ API parity test, please see:\n"
            "NOTE [How to check NN module / functional API parity between Python and C++ frontends]. \n"
            "If not, please add `test_cpp_api_parity=False` to the test params dict and file an issue about this."
        )

    return TorchNNModuleTestParams(
        module_name=module_name,
        module_variant_name=module_variant_name,
        test_instance=test_instance,
        cpp_constructor_args=test_params_dict.get("cpp_constructor_args", ""),
        arg_dict=compute_arg_dict(test_params_dict, test_instance),
        has_parity=test_params_dict.get("has_parity", True),
        device=device,
        cpp_tmp_folder=tempfile.mkdtemp(),
    )


def write_test_to_test_class(
    unit_test_class, test_params_dict, test_instance_class, parity_table, devices
):
    assert not is_torch_nn_functional_test(test_params_dict)

    module_name = compute_module_name(test_params_dict)

    assert hasattr(torch.nn, module_name), (
        f"`torch.nn` doesn't have module `{module_name}`. "
        "If you are adding a new test, please set `fullname` using format `ModuleName_desc` "
        f"or set `module_name` using format `ModuleName` in the module test dict:\n{pprint.pformat(test_params_dict)}"
    )

    module_full_name = "torch::nn::" + module_name

    assert module_full_name in parity_table["torch::nn"], (
        f"Please add `{module_full_name}` entry to `torch::nn` section of `test/cpp_api_parity/parity-tracker.md`. "
        f"(Discovered while processing\n{pprint.pformat(test_params_dict)}.)"
    )

    for device in devices:
        test_params = process_test_params_for_module(
            test_params_dict=test_params_dict,
            device=device,
            test_instance_class=test_instance_class,
        )
        try_remove_folder(test_params.cpp_tmp_folder)
        unit_test_name = f"test_torch_nn_{test_params.module_variant_name}"
        unit_test_class.module_test_params_map[unit_test_name] = test_params

        def test_fn(self):
            test_forward_backward(
                unit_test_class=self,
                test_params=unit_test_class.module_test_params_map[
                    self._testMethodName
                ],
            )

        test_fn = decorate_test_fn(
            test_fn=test_fn,
            test_cuda=test_params_dict.get("test_cuda", True),
            has_impl_parity=parity_table["torch::nn"][module_full_name][0]
            and test_params_dict.get("has_parity", True),
            device=device,
        )

        add_test(unit_test_class, unit_test_name, test_fn)


def generate_test_cpp_sources(test_params, template):
    device = test_params.device

    cpp_constructor_args = test_params.cpp_constructor_args
    if cpp_constructor_args != "":
        cpp_constructor_args = f"({cpp_constructor_args})"

    (
        cpp_args_construction_stmts,
        cpp_forward_args_symbols,
    ) = compute_cpp_args_construction_stmts_and_forward_arg_symbols(test_params)

    test_cpp_sources = template.substitute(
        module_variant_name=test_params.module_variant_name,
        module_qualified_name=f"torch::nn::{test_params.module_name}",
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
    for test_params in unit_test_class.module_test_params_map.values():
        cpp_sources += generate_test_cpp_sources(
            test_params=test_params, template=TORCH_NN_MODULE_TEST_FORWARD_BACKWARD
        )
        functions.append(f"{test_params.module_variant_name}_test_forward_backward")
    if print_cpp_source:
        print(cpp_sources)

    cpp_module = compile_cpp_code_inline(
        name="module_impl_check", cpp_sources=cpp_sources, functions=functions
    )
    unit_test_class.module_impl_check_cpp_module = cpp_module
