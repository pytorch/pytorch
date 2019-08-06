import os
import shutil
import tempfile
from string import Template

import torch
import common_utils as common
from common_nn import module_tests
import torch.utils.cpp_extension
from cpp_api_parity import sample_module

torch_nn_has_parity = ['Linear']


class TestCppApiParity(common.TestCase):
    def setUp(self):
        default_build_root = torch.utils.cpp_extension.get_default_build_root()
        if os.path.exists(default_build_root):
            shutil.rmtree(default_build_root)

    def _test_torch_nn_modules(
            self,
            test_suite_name,
            module_names,
            module_tests,
            python_module_class=None,
            cpp_namespace='torch::nn::'):
        torch_nn_test_methods = ['init', 'forward']

        TORCH_NN_MODULE_COMMON_TEST_HARNESS = """\n
const char * const parity_test_error_msg = "Parity test failed";

void test_module_state_equality(std::shared_ptr<torch::nn::Module> m1, std::shared_ptr<torch::nn::Module> m2) {
  auto params1 = m1->named_parameters();
  auto params2 = m2->named_parameters();
  TORCH_CHECK(
    params1.size() == params2.size(),
    parity_test_error_msg, ": ", "# of parameters doesn't match");
  for (auto& param : params1)
    TORCH_CHECK(
      param.value().allclose(params2[param.key()]),
      parity_test_error_msg, ": ", "value of `", param.key(), "` doesn't match");

  auto buffers1 = m1->named_buffers();
  auto buffers2 = m2->named_buffers();
  TORCH_CHECK(
    buffers1.size() == buffers2.size(),
    parity_test_error_msg, ": ", "# of buffers doesn't match");
  for (auto& buffer : buffers1)
    TORCH_CHECK(
      buffer.value().allclose(buffers2[buffer.key()]),
      parity_test_error_msg, ": ", "value of `", buffer.key(), "` doesn't match");
}
"""

        TORCH_NN_MODULE_WRAPPER = Template("""\n
void ${module_variant_name}_test_init(const std::string& saved_module_path) {
  // NOTE: `m_init_by_cpp` must be constructed before `m_init_by_python`,
  // because we want `m_init_by_cpp`'s initialization to use a clean random number
  // generator state (which we reset before calling this function).
  ${module_qualified_name} m_init_by_cpp${cpp_constructor_args};

  ${module_qualified_name} m_init_by_python${cpp_constructor_args};
  torch::load(m_init_by_python, saved_module_path);

  test_module_state_equality(m_init_by_cpp.ptr(), m_init_by_python.ptr());
}

void ${module_variant_name}_test_forward(
    const std::string& saved_module_path,
    torch::Tensor input,
    torch::Tensor python_output) {
  ${module_qualified_name} module${cpp_constructor_args};
  torch::load(module, saved_module_path);
  TORCH_CHECK(
    module(input).allclose(python_output),
    parity_test_error_msg, ": forward output doesn't match");
}
""")

        def process_test_params(test_params, python_module_class, cpp_namespace):
            module_name = test_params['module_name']
            desc = test_params.get('desc', None)
            test_params['module_variant_name'] = module_name + (('_' + desc) if desc else '')
            test_params['python_constructor_args'] = test_params.get('constructor_args')
            test_params['cpp_constructor_args'] = test_params.get('cpp_constructor_args')
            test_params['cpp_namespace'] = cpp_namespace
            test_params['cpp_source'] = test_params.get('cpp_source', '')
            test_params['expect_error'] = test_params.get('expect_error', False)
            if not python_module_class:
                python_module_class = getattr(torch.nn, module_name)
            test_params['python_module_class'] = python_module_class
            return test_params

        test_params_map = {}
        for test_params in module_tests:
            test_params = process_test_params(test_params, python_module_class, cpp_namespace)
            module_name = test_params['module_name']
            if module_name not in test_params_map:
                test_params_map[module_name] = []
            test_params_map[module_name].append(test_params)

        # Generate C++ code for each test case
        cpp_source = TORCH_NN_MODULE_COMMON_TEST_HARNESS + test_params['cpp_source']
        functions = []
        for module_name in module_names:
            for test_params in test_params_map[module_name]:
                cpp_source += TORCH_NN_MODULE_WRAPPER.substitute(
                    module_variant_name=test_params['module_variant_name'],
                    module_qualified_name=test_params['cpp_namespace'] + test_params['module_name'],
                    cpp_constructor_args=test_params['cpp_constructor_args'])
                for method in torch_nn_test_methods:
                    functions.append(test_params['module_variant_name'] + '_test_' + method)

        # Just-in-time compile the C++ test code
        cpp_module = torch.utils.cpp_extension.load_inline(
            name=test_suite_name,
            cpp_sources=cpp_source,
            functions=functions,
            verbose=True,
        )

        def test_method(method_name, test_params):
            expect_error = test_params['expect_error']
            input_size = test_params['input_size']
            python_module_class = test_params['python_module_class']
            python_constructor_args = test_params['python_constructor_args']
            module_variant_name = test_params['module_variant_name']
            test_name = module_variant_name + '_test_' + method_name
            example_input = torch.randn(input_size)
            with common.freeze_rng_state():
                torch.manual_seed(2)
                module = python_module_class(*python_constructor_args)
                if method_name == 'forward':
                    python_output = module(example_input)
                # We use JIT tracing to transfer Python module state to C++
                traced_script_module = torch.jit.trace(module, example_input)
                # Ideally we would like to not have to manually delete the file, but NamedTemporaryFile
                # opens the file, and it cannot be opened multiple times in Windows. To support Windows,
                # close the file after creation and try to remove it manually
                f = tempfile.NamedTemporaryFile(delete=False)
                try:
                    f.close()
                    traced_script_module.save(f.name)
                    torch.manual_seed(2)
                    if method_name == 'init':
                        args = (f.name, )
                    elif method_name == 'forward':
                        args = (f.name, example_input, python_output)
                    if expect_error:
                        with self.assertRaisesRegex(RuntimeError, 'Parity test failed'):
                            getattr(cpp_module, test_name)(*args)
                    else:
                        getattr(cpp_module, test_name)(*args)
                finally:
                    os.unlink(f.name)

        for module_name in module_names:
            for test_params in test_params_map[module_name]:
                for method in torch_nn_test_methods:
                    test_method(method, test_params)

    def test_sample_module(self):
        self._test_torch_nn_modules(
            test_suite_name='test_sample_module',
            module_names=['SampleModule'],
            module_tests=sample_module.module_tests,
            python_module_class=sample_module.SampleModule,
            cpp_namespace='')

    def test_torch_nn(self):
        self._test_torch_nn_modules(
            test_suite_name='test_torch_nn',
            module_names=torch_nn_has_parity,
            module_tests=module_tests)


if __name__ == "__main__":
    common.run_tests()
