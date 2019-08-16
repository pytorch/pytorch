import os
import tempfile
from string import Template
import copy
import unittest
import warnings
import inspect

import torch
import common_utils as common
import common_nn
from common_cuda import TEST_CUDA
import torch.utils.cpp_extension
from cpp_api_parity import sample_module, torch_nn_modules, TorchNNTestParams


torch_nn_has_parity = set([
    'SampleModule',
    'Linear',
])

TORCH_NN_MODULE_COMMON_TEST_HARNESS = """\n
#include <torch/script.h>

const char * const parity_test_error_msg = "Parity test failed";

/*
void test_module_state_equality(std::shared_ptr<torch::nn::Module> m1, std::shared_ptr<torch::nn::Module> m2) {
  auto params1 = m1->named_parameters();
  auto params2 = m2->named_parameters();
  TORCH_CHECK(
    params1.size() == params2.size(),
    parity_test_error_msg, ": ", "# of parameters doesn't match");
  for (auto& param : params1) {
    TORCH_CHECK(
      param->sizes().vec() == params2[param.key()].sizes().vec() &&
      param->allclose(params2[param.key()]),
      parity_test_error_msg, ": ", "sizes or value of `", param.key(), "` doesn't match");
  }

  auto buffers1 = m1->named_buffers();
  auto buffers2 = m2->named_buffers();
  TORCH_CHECK(
    buffers1.size() == buffers2.size(),
    parity_test_error_msg, ": ", "# of buffers doesn't match");
  for (auto& buffer : buffers1) {
    TORCH_CHECK(
      buffer->sizes().vec() == buffers2[buffer.key()].sizes().vec() &&
      buffer->allclose(buffers2[buffer.key()]),
      parity_test_error_msg, ": ", "sizes or value of `", buffer.key(), "` doesn't match");
  }
}
*/

bool check_tensor_equality(const torch::Tensor& tensor1, const torch::Tensor& tensor2) {
  return tensor1.sizes().vec() == tensor2.sizes().vec() && tensor1.allclose(tensor2);
}

// yf225 TODO: comment on data type support
/*
  _(Tensor) \
  _(Double) \
  _(Int) \
  _(Bool) \
  _(String) \
*/
bool check_ivalue_equality(const c10::IValue& ivalue1, const c10::IValue& ivalue2) {
  if (ivalue1.tagKind() != ivalue2.tagKind()) {
    AT_ERROR("Value type mismatch: ", "ivalue1: ", ivalue1.tagKind(), ", ivalue2: ", ivalue2.tagKind());
  }
  if (ivalue1.isInt()) {
    return ivalue1.toInt() == ivalue2.toInt();
  } else if (ivalue1.isDouble()) {
    return ivalue1.toDouble() == ivalue2.toDouble();
  } else if (ivalue1.isBool()) {
    return ivalue1.toBool() == ivalue2.toBool();
  } else if (ivalue1.isString()) {
    return ivalue1.toString() == ivalue2.toString();
  } else if (ivalue1.isTensor()) {
    return check_tensor_equality(ivalue1.toTensor(), ivalue2.toTensor());
  } else {
    AT_ERROR("Unsupported value type: ", ivalue1.tagKind());
  }
}
"""

CHECK_MODULE_PARAM_EQUALITY = Template("""\
TORCH_CHECK(check_tensor_equality(${script_module_prefix}.get_parameter("${param_name}"), ${cpp_module_prefix}->${param_name}), parity_test_error_msg);
""")

CHECK_MODULE_BUFFER_EQUALITY = Template("""\
TORCH_CHECK(check_tensor_equality(${script_module_prefix}.get_buffer("${buffer_name}"), ${cpp_module_prefix}->${buffer_name}), parity_test_error_msg);
""")

CHECK_MODULE_ATTR_EQUALITY = Template("""\
TORCH_CHECK(check_ivalue_equality(${script_module_prefix}.get_attribute("${attr_name}"), c10::IValue(${cpp_module_prefix}->${attr_name})), parity_test_error_msg);
""")

TORCH_NN_MODULE_TEST_INIT = Template("""\n
void ${module_variant_name}_test_init(
    const std::string& saved_module_path,
    const std::string& device) {
  torch::jit::script::Module m_init_by_python = torch::jit::load(saved_module_path);

  torch::manual_seed(2);
  ${module_qualified_name} m_init_by_cpp${cpp_constructor_args};
  m_init_by_cpp->to(device);

  /*
  std::cout << "number of slots in python module: " << m_init_by_python.get_slots().size() << std::endl;

  for (const auto& slot : m_init_by_python.get_slots()) {
    std::cout << slot.name() << std::endl;
  }
  */

  /*
  TORCH_CHECK(m_init_by_python.get_parameter("param").allclose(m_init_by_cpp->param), parity_test_error_msg);
  TORCH_CHECK(m_init_by_python.get_buffer("buffer").allclose(m_init_by_cpp->buffer), parity_test_error_msg);
  // yf225 TODO: we have to generate this part from Python
  // yf225 TODO: we probably don't need to check all types now, let's just check int, float/double, bool, tensor for now, and throw error if we encounter other types
  // yf225 TODO: we need to check submodules recursively
  TORCH_CHECK(m_init_by_python.get_attribute("attr").toInt() == m_init_by_cpp->attr, parity_test_error_msg);
  */

  /* yf225 TODO: we need to check these:
  1. param / buffer / attr names and values parity (TODO: we need to check submodules recursively as well!)
    1. Need a way to list out those names in the test
        1. For buffers, get them from get_attributes() of each module
        2. For attrs, use `module.__dict__.keys()` and filter out unrelated names to find all unregistered attributes, and register them manually using _register_attribute
          - Make sure to collect all attributes recursively
          - Use traced_script_module._c._register_attribute("attr", torch.jit.annotations.ann_to_type(type(module.attr)), module.attr)
          - Can we convert all the C++ values to IValue, so that we don't need to worry about their actual type when comparing?
        2. For parameters, get them from get_parameters() of each module
  */

  // test_module_state_equality(m_init_by_python.ptr(), m_init_by_cpp.ptr());

  ${extra_stmts}
}
""")

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

  ${extra_stmts}
}
""")

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

  ${extra_stmts}
}
""")

class TestCppApiParity(common.TestCase):
    # yf225 TODO: test that constructor arg names match on both sides (chech name match only, no need to check value match. But need to translate default Python values to C++ values)
    # (Can we convert them to IValue then convert from IValue to unpacked value in C++? We should generate the `options` statement)
    # yf225 TODO: For things like FanMode, we should support dual mode (string / enum), for legacy support
    def _test_torch_nn_module_ctor_args(self, module_name):
        # yf225 TODO
        python_module_class = getattr(torch.nn, module_name)
        print(inspect.getfullargspec(python_module_class.__init__))

    def _test_torch_nn_module_variant(self, test_params):
        torch_nn_test_methods = ['init', 'forward', 'backward']

        def setup_init_test(device, python_module_class, python_constructor_args):
            torch.manual_seed(2)
            module = python_module_class(*python_constructor_args).to(device)
            return [module], device

        def setup_forward_test(device, python_module_class, python_constructor_args, example_inputs):
            torch.manual_seed(2)
            module = python_module_class(*python_constructor_args).to(device)
            python_output = module(*example_inputs)
            return [module], device, python_output, example_inputs

        def setup_backward_test(device, python_module_class, python_constructor_args, example_inputs):
            torch.manual_seed(2)
            module = python_module_class(*python_constructor_args).to(device)
            python_output = module(*example_inputs)
            python_output.sum().backward()
            # JIT tracing does not save a module's parameters' gradients into ScriptModule.
            # Instead, we create another module `grad_module` with the same structure as `module`,
            # and use `grad_module`'s parameters to save `module`'s corresponding parameters'
            # gradients. Then, we trace both `module` and `grad_module`, serialize them and
            # pass them into C++ for parity testing.
            grad_module = copy.deepcopy(module)
            for param, grad_param in zip(module.parameters(), grad_module.parameters()):
                if param.grad is not None:
                    grad_param.data = param.grad
            return [module, grad_module], device, example_inputs

        # yf225 TODO: maybe a better name?
        # yf225 TODO: just generate something like (in C++) `module.get_module("submodule").get_XX(some_field_name) .. == .. cpp_module->submodule->some_field_name
        def generate_attr_checks_recursively(module, stmts, script_module_prefix='m_init_by_python', cpp_module_prefix='m_init_by_cpp'):
            for name, sub_module in module.named_children():
                sub_script_module_prefix = '{}.get_module("{}")'.format(script_module_prefix, name)
                sub_cpp_module_prefix = '{}->{}'.format(cpp_module_prefix, name)
                generate_attr_checks_recursively(sub_module, stmts, sub_script_module_prefix, sub_cpp_module_prefix)
            for name, param in module._parameters.items():
                stmts += CHECK_MODULE_PARAM_EQUALITY.substitute(
                    script_module_prefix=script_module_prefix,
                    cpp_module_prefix=cpp_module_prefix,
                    param_name=name)
            for name, buffer in module._buffers.items():
                stmts += CHECK_MODULE_BUFFER_EQUALITY.substitute(
                    script_module_prefix=script_module_prefix,
                    cpp_module_prefix=cpp_module_prefix,
                    buffer_name=name)
            for name, attr in module.__dict__.items():
                if name not in ['_backend', '_parameters', '_buffers', '_backward_hooks', '_forward_hooks', '_forward_pre_hooks', '_state_dict_hooks', '_load_state_dict_pre_hooks', '_modules', 'training', 'has_parity']:
                    stmts += CHECK_MODULE_ATTR_EQUALITY.substitute(
                        script_module_prefix=script_module_prefix,
                        cpp_module_prefix=cpp_module_prefix,
                        attr_name=name)

        def generate_and_compile_cpp_test_functions(test_init_module, test_params):
            cpp_source = TORCH_NN_MODULE_COMMON_TEST_HARNESS + test_params.cpp_source
            functions = []
            cpp_forward_arg_declarations = test_params.cpp_forward_arg_declarations
            input_arg_declarations = ',\n'.join([arg_type + ' ' + arg_name for arg_type, arg_name in cpp_forward_arg_declarations])
            input_args = ',\n'.join([arg_name for arg_type, arg_name in cpp_forward_arg_declarations])
            for template in [TORCH_NN_MODULE_TEST_INIT, TORCH_NN_MODULE_TEST_FORWARD, TORCH_NN_MODULE_TEST_BACKWARD]:
                extra_stmts = ''
                if template == TORCH_NN_MODULE_TEST_INIT:
                    extra_stmt_list = []
                    generate_attr_checks_recursively(test_init_module, extra_stmt_list)
                    extra_stmts = ''.join(extra_stmt_list)
                cpp_source += template.substitute(
                    module_variant_name=test_params.module_variant_name,
                    module_qualified_name='torch::nn::' + test_params.module_name,
                    cpp_constructor_args=test_params.cpp_constructor_args,
                    input_arg_declarations=input_arg_declarations,
                    input_args=input_args,
                    extra_stmts=extra_stmts)
            for method in torch_nn_test_methods:
                functions.append(test_params.module_variant_name + '_test_' + method)

            # Just-in-time compile the C++ test code
            cpp_module = torch.utils.cpp_extension.load_inline(
                name=test_params.module_variant_name,
                cpp_sources=cpp_source,
                functions=functions,
                verbose=False,
            )
            return cpp_module

        # yf225 TODO: explain why we need to do this
        def register_attrs_recursively(module, script_module):
            for sub_module, sub_script_module in zip(module.children(), script_module.children()):
                register_attrs_recursively(sub_module, sub_script_module)

            for key, value in module.__dict__.items():
                if key not in ['_backend', '_parameters', '_buffers', '_backward_hooks', '_forward_hooks', '_forward_pre_hooks', '_state_dict_hooks', '_load_state_dict_pre_hooks', '_modules', 'training', 'has_parity']:
                    script_module._c._register_attribute(key, torch.jit.annotations.ann_to_type(type(value)), value)

        def serialize_module_into_file(module, example_inputs):
            # We use JIT tracing to serialize Python module state, so that we can load it into C++
            traced_script_module = torch.jit.trace(module, example_inputs)
            register_attrs_recursively(module, traced_script_module)
            module_file = tempfile.NamedTemporaryFile(delete=False)
            traced_script_module.save(module_file.name)
            module_file.close()
            return module_file.name

        def test_methods(test_params):
            device = test_params.device
            input_size = test_params.input_size
            input_fn = test_params.input_fn
            python_module_class = test_params.python_module_class
            python_constructor_args = test_params.python_constructor_args
            module_variant_name = test_params.module_variant_name

            if input_size:
                example_inputs = [torch.randn(input_size)]
            elif input_fn:
                example_inputs = list(input_fn())
            else:
                raise RuntimeError("Missing `input_size` or `input_fn` for {}".format(module_variant_name))
            example_inputs = [x.to(device) for x in example_inputs]

            args_map = {}

            for method_name in torch_nn_test_methods:
                if method_name == 'init':
                    args_map[method_name] = setup_init_test(device, python_module_class, python_constructor_args)
                elif method_name == 'forward':
                    args_map[method_name] = setup_forward_test(device, python_module_class, python_constructor_args, example_inputs)
                elif method_name == 'backward':
                    args_map[method_name] = setup_backward_test(device, python_module_class, python_constructor_args, example_inputs)
                else:
                    raise RuntimeError("{} is not a supported method to test".format(method_name))

            test_init_module = args_map['init'][0][0]
            cpp_module = generate_and_compile_cpp_test_functions(test_init_module, test_params)

            for method_name in torch_nn_test_methods:
                args = args_map[method_name]
                modules = args[0]
                module_file_names = [serialize_module_into_file(module, example_inputs) for module in modules]

                cpp_args = module_file_names[:]
                for arg in args[1:]:
                    if isinstance(arg, list):
                        cpp_args += arg
                    else:
                        cpp_args.append(arg)

                try:
                    cpp_test_name = module_variant_name + '_test_' + method_name
                    cpp_test_fn = getattr(cpp_module, cpp_test_name)
                    if test_params.expect_parity_error:
                        with self.assertRaisesRegex(RuntimeError, "Parity test failed"):
                            cpp_test_fn(*cpp_args)
                    else:
                        cpp_test_fn(*cpp_args)
                finally:
                    # Ideally we would like to not have to manually delete the file, but NamedTemporaryFile
                    # opens the file, and it cannot be opened multiple times in Windows. To support Windows,
                    # we close the file after creation and try to remove it manually.
                    for module_file_name in module_file_names:
                        try:
                            os.remove(module_file_name)
                        except OSError as e:
                            warnings.warn("Unable to remove {}, got error: {}".format(module_file_name, str(e)))

        test_methods(test_params)


def _process_test_params(test_params_dict, module_metadata, device):
    module_name = test_params_dict.get('module_name')
    desc = test_params_dict.get('desc', None)
    return TorchNNTestParams(
        module_name=module_name,
        module_variant_name=module_name + (('_' + desc) if desc else '') + (('_' + device) if device != 'cpu' else ''),
        python_constructor_args=test_params_dict.get('constructor_args'),
        cpp_constructor_args=test_params_dict.get('cpp_constructor_args'),
        input_size=test_params_dict.get('input_size', None),
        input_fn=test_params_dict.get('input_fn', None),
        expect_parity_error=test_params_dict.get('expect_parity_error', False),
        cpp_forward_arg_declarations=module_metadata.get('cpp_forward_arg_declarations'),
        python_module_class=getattr(torch.nn, module_name),
        cpp_source=module_metadata.get('cpp_source', ''),
        device=device,
    )


def add_test(test_name, test_fn):
    if hasattr(TestCppApiParity, test_name):
        raise RuntimeError("Found two tests with the same name: " + test_name)
    setattr(TestCppApiParity, test_name, test_fn)

torch_nn_test_params_map = {}

torch_nn_modules.module_metadata_map['SampleModule'] = sample_module.module_metadata

torch_nn_module_names = set()

# for test_params_dict in sample_module.module_tests + common_nn.module_tests: 
for test_params_dict in sample_module.module_tests:  # yf225 TODO: run on all nn modules
    module_name = test_params_dict.get('module_name')
    torch_nn_module_names.add(module_name)
    if module_name in torch_nn_has_parity:
        module_metadata = torch_nn_modules.module_metadata_map[module_name]
        for device in ['cpu', 'cuda']:
            test_params = _process_test_params(
                test_params_dict=test_params_dict,
                module_metadata=module_metadata,
                device=device)
            test_name = 'test_torch_nn_' + test_params.module_variant_name
            torch_nn_test_params_map[test_name] = test_params

            def test_fn(self):
                self._test_torch_nn_module_variant(test_params=torch_nn_test_params_map[self._testMethodName])

            if device == 'cuda':
                test_fn = unittest.skipIf(not TEST_CUDA, "CUDA unavailable")(test_fn)
            add_test(test_name, test_fn)

for module_name in sorted(list(torch_nn_module_names)):
    ctor_args_test_name = 'test_torch_nn_{}_ctor_args'.format(module_name)

    def ctor_args_test(self):
        self._test_torch_nn_module_ctor_args(module_name=self._testMethodName.strip('test_torch_nn_').strip('_ctor_args'))

    add_test(ctor_args_test_name, ctor_args_test)


if __name__ == "__main__":
    common.run_tests()
