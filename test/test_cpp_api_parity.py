import os
import shutil
import tempfile
from string import Template
import copy
import unittest

import torch
import common_utils as common
import common_nn
from common_cuda import TEST_CUDA
import torch.utils.cpp_extension
from cpp_api_parity import sample_module, torch_nn_modules


torch_nn_has_parity = set([
    'SampleModule',
    'Linear',
])


class TestCppApiParity(common.TestCase):
    def _test_torch_nn_module(self, test_params):
        torch_nn_test_methods = ['init', 'forward', 'backward']

        TORCH_NN_MODULE_COMMON_TEST_HARNESS = """\n
#include <torch/script.h>

const char * const parity_test_error_msg = "Parity test failed";

void test_module_state_equality(std::shared_ptr<torch::nn::Module> m1, std::shared_ptr<torch::nn::Module> m2) {
  auto params1 = m1->named_parameters();
  auto params2 = m2->named_parameters();
  TORCH_CHECK(
    params1.size() == params2.size(),
    parity_test_error_msg, ": ", "# of parameters doesn't match");
  for (auto& param : params1) {
    TORCH_CHECK(
      param->sizes().vec() == params2[param.key()].sizes().vec(),
      parity_test_error_msg, ": ", "sizes of `", param.key(), "` doesn't match");
    TORCH_CHECK(
      param->allclose(params2[param.key()]),
      parity_test_error_msg, ": ", "value of `", param.key(), "` doesn't match");
  }

  auto buffers1 = m1->named_buffers();
  auto buffers2 = m2->named_buffers();
  TORCH_CHECK(
    buffers1.size() == buffers2.size(),
    parity_test_error_msg, ": ", "# of buffers doesn't match");
  for (auto& buffer : buffers1) {
    TORCH_CHECK(
      buffer->sizes().vec() == buffers2[buffer.key()].sizes().vec(),
      parity_test_error_msg, ": ", "sizes of `", buffer.key(), "` doesn't match");
    TORCH_CHECK(
      buffer->allclose(buffers2[buffer.key()]),
      parity_test_error_msg, ": ", "value of `", buffer.key(), "` doesn't match");
  }
}
"""
        def test_init(device, python_module_class, python_constructor_args):
            torch.manual_seed(2)
            module = python_module_class(*python_constructor_args)
            module = module.to(device)
            return [module], device

        TORCH_NN_MODULE_TEST_INIT = Template("""\n
void ${module_variant_name}_test_init(
    const std::string& saved_module_path,
    const std::string& device) {
  ${module_qualified_name} m_init_by_python${cpp_constructor_args};
  torch::load(m_init_by_python, saved_module_path);

  torch::manual_seed(2);
  ${module_qualified_name} m_init_by_cpp${cpp_constructor_args};
  m_init_by_cpp->to(device);

  test_module_state_equality(m_init_by_cpp.ptr(), m_init_by_python.ptr());
}
""")

        def test_forward(device, python_module_class, python_constructor_args, example_inputs):
            torch.manual_seed(2)
            module = python_module_class(*python_constructor_args)
            module = module.to(device)
            python_output = module(*example_inputs)
            return [module], device, python_output, example_inputs

        TORCH_NN_MODULE_TEST_FORWARD = Template("""\n
void ${module_variant_name}_test_forward(
    const std::string& saved_module_path,
    const std::string& device,
    torch::Tensor python_output,
    ${input_arg_declarations}) {
  torch::manual_seed(2);
  ${module_qualified_name} module${cpp_constructor_args};
  torch::load(module, saved_module_path);
  module->to(device);

  auto cpp_output = module(${input_args});

  TORCH_CHECK(
    cpp_output.sizes().vec() == python_output.sizes().vec() &&
    cpp_output.allclose(python_output),
    parity_test_error_msg, ": forward output doesn't match");
}
""")

        def test_backward(device, python_module_class, python_constructor_args, example_inputs):
            torch.manual_seed(2)
            module = python_module_class(*python_constructor_args)
            module = module.to(device)
            python_output = module(*example_inputs)
            python_output.sum().backward()
            # JIT tracing does not save a module's parameters' gradients into ScriptModule.
            # Instead, we create another module `grad_module` with the same structure as `module`,
            # and use `grad_module`'s parameters to save `module`'s corresponding parameters'
            # gradients. Then, we trace both `module` and `grad_module`, serialize them and
            # pass them into C++ for parity testing.
            grad_module = copy.deepcopy(module)
            for param, grad_param in zip(module.parameters(), grad_module.parameters()):
                grad_param.data = param.grad
            return [module, grad_module], device, example_inputs

        TORCH_NN_MODULE_TEST_BACKWARD = Template("""\n
void ${module_variant_name}_test_backward(
    const std::string& saved_module_path,
    const std::string& saved_grad_module_path,
    const std::string& device,
    ${input_arg_declarations}) {
  ${module_qualified_name} grad_module${cpp_constructor_args};
  torch::load(grad_module, saved_grad_module_path);

  torch::manual_seed(2);
  ${module_qualified_name} module${cpp_constructor_args};
  torch::load(module, saved_module_path);
  module->to(device);

  auto cpp_output = module(${input_args});
  cpp_output.sum().backward();

  for (size_t i = 0; i < module->parameters().size(); i++) {
    auto named_param = module->named_parameters()[i];
    auto grad = grad_module->parameters()[i];
    TORCH_CHECK(
      named_param->grad().allclose(grad),
      parity_test_error_msg, ": ", "gradient value of `", named_param.key(), "` doesn't match");
  }
}
""")

        # Generate C++ code for this test case
        cpp_source = TORCH_NN_MODULE_COMMON_TEST_HARNESS + test_params['cpp_source']
        cpp_namespace = test_params['cpp_namespace']
        functions = []
        cpp_forward_arg_declarations = test_params['cpp_forward_arg_declarations']
        input_arg_declarations = ',\n'.join([arg_type + ' ' + arg_name for arg_type, arg_name in cpp_forward_arg_declarations])
        input_args = ',\n'.join([arg_name for arg_type, arg_name in cpp_forward_arg_declarations])
        for template in [TORCH_NN_MODULE_TEST_INIT, TORCH_NN_MODULE_TEST_FORWARD, TORCH_NN_MODULE_TEST_BACKWARD]:
            cpp_source += template.substitute(
                module_variant_name=test_params['module_variant_name'],
                module_qualified_name=cpp_namespace + test_params['module_name'],
                cpp_constructor_args=test_params['cpp_constructor_args'],
                input_arg_declarations=input_arg_declarations,
                input_args=input_args)
        for method in torch_nn_test_methods:
            functions.append(test_params['module_variant_name'] + '_test_' + method)

        # Just-in-time compile the C++ test code
        cpp_module = torch.utils.cpp_extension.load_inline(
            name=test_params['module_variant_name'],
            cpp_sources=cpp_source,
            functions=functions,
            verbose=False,
        )

        def test_method(method_name, test_params):
            device = test_params['device']
            input_size = test_params.get('input_size', None)
            input_fn = test_params.get('input_fn', None)
            python_module_class = test_params['python_module_class']
            python_constructor_args = test_params['python_constructor_args']
            module_variant_name = test_params['module_variant_name']
            cpp_test_name = module_variant_name + '_test_' + method_name
            if input_size:
                example_inputs = (torch.randn(input_size), )
            elif input_fn:
                example_inputs = input_fn()
            else:
                raise RuntimeError('Missing `input_size` or `input_fn` for {}'.format(module_variant_name))
            example_inputs = tuple([x.to(device) for x in example_inputs])
            if method_name == 'init':
                args = test_init(device, python_module_class, python_constructor_args)
            elif method_name == 'forward':
                args = test_forward(device, python_module_class, python_constructor_args, example_inputs)
            elif method_name == 'backward':
                args = test_backward(device, python_module_class, python_constructor_args, example_inputs)
            # We use JIT tracing to transfer Python module state to C++
            modules = args[0]
            module_file_names = []
            for module in modules:
                traced_script_module = torch.jit.trace(module, example_inputs)
                module_file = tempfile.NamedTemporaryFile(delete=False)
                traced_script_module.save(module_file.name)
                module_file.close()
                module_file_names.append(module_file.name)
            
            cpp_args = []
            for module_file_name in module_file_names:
                cpp_args.append(module_file_name)
            for arg in args[1:]:
                if isinstance(arg, tuple):
                    cpp_args += list(arg)
                else:
                    cpp_args.append(arg)
            try:
                getattr(cpp_module, cpp_test_name)(*cpp_args)
            finally:
                # Ideally we would like to not have to manually delete the file, but NamedTemporaryFile
                # opens the file, and it cannot be opened multiple times in Windows. To support Windows,
                # we close the file after creation and try to remove it manually.
                for module_file_name in module_file_names:
                    os.unlink(module_file_name)

        for method in torch_nn_test_methods:
            test_method(method, test_params)


def _process_test_params(test_params, module_metadata, device):
    module_name = test_params.get('module_name')
    desc = test_params.get('desc', None)
    test_params['module_variant_name'] = module_name + (('_' + desc) if desc else '') + '_' + device
    test_params['python_constructor_args'] = test_params.get('constructor_args')
    test_params['cpp_constructor_args'] = test_params.get('cpp_constructor_args')
    test_params['expect_parity_error'] = test_params.get('expect_parity_error', False)
    test_params['cpp_forward_arg_declarations'] = module_metadata.get('cpp_forward_arg_declarations')
    python_module_class = module_metadata.get('python_module_class', None)
    if not python_module_class:
        python_module_class = getattr(torch.nn, module_name)
    test_params['python_module_class'] = python_module_class
    test_params['cpp_namespace'] = module_metadata.get('cpp_namespace', 'torch::nn::')
    test_params['cpp_source'] = module_metadata.get('cpp_source', '')
    test_params['device'] = device
    return test_params


def add_test(test_name, test_fn):
    if hasattr(TestCppApiParity, test_name):
        raise RuntimeError('Found two tests with the same name: ' + test_name)
    setattr(TestCppApiParity, test_name, test_fn)

torch_nn_test_params_map = {}

for test_params in sample_module.module_tests + common_nn.module_tests:
    module_name = test_params.get('module_name')
    desc = test_params.get('desc', None)
    if module_name in torch_nn_has_parity:
        if module_name == 'SampleModule':
            module_metadata = sample_module.module_metadata
        else:
            module_metadata = torch_nn_modules.module_metadata_map[module_name]
        for device in ['cpu', 'cuda']:
            test_name = 'test_torch_nn_' + module_name + (('_' + desc) if desc else '') + (('_' + device) if device != 'cpu' else '')
            test_params = _process_test_params(
                test_params=test_params,
                module_metadata=module_metadata,
                device=device)
            torch_nn_test_params_map[test_name] = test_params
            def test_fn(self):
                self._test_torch_nn_module(test_params=torch_nn_test_params_map[self._testMethodName])
            if device == 'cuda':
                test_fn = unittest.skipIf(not TEST_CUDA, "CUDA unavailable")(test_fn)
            expect_parity_error = test_params.get('expect_parity_error')
            if expect_parity_error:
                test_fn = unittest.expectedFailure(test_fn)
            add_test(test_name, test_fn)


if __name__ == "__main__":
    common.run_tests()
