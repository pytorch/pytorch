#include <torch/csrc/utils/nested.h>
#include <torch/csrc/utils/pycfunction_helpers.h>
#include <torch/csrc/utils/python_arg_parser.h>
#include <torch/torch.h>

namespace torch::autograd {

static PyObject* THPVariable_nested_tensor(
    PyObject* /*self*/,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
      "nested_tensor(PyObject* data, *, ScalarType dtype=None, Device? device=None, bool pin_memory=False, bool requires_grad=False)",
  });

  constexpr int ctor_num_args = 5;
  ParsedArgs<ctor_num_args> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  jit::tracer::warn(
      "torch.nested.nested_tensor", jit::tracer::WARN_CONSTRUCTOR);
  return THPVariable_Wrap(torch::utils::nested_tensor_ctor(
      torch::tensors::get_default_dispatch_key(),
      torch::tensors::get_default_scalar_type(),
      r));
  END_HANDLE_TH_ERRORS
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
static PyMethodDef nested_functions_manual[] = {
    {"nested_tensor",
     castPyCFunctionWithKeywords(THPVariable_nested_tensor),
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
};

PyMethodDef* get_nested_functions_manual() {
  return nested_functions_manual;
}

} // namespace torch::autograd
