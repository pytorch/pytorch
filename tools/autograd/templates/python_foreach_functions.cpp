#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// ${generated_comment}

#include "torch/csrc/Device.h"
#include "torch/csrc/DynamicTypes.h"
#include "torch/csrc/Exceptions.h"
#include "torch/csrc/autograd/generated/python_return_types.h"
#include "torch/csrc/autograd/generated/variable_factories.h"
#include "torch/csrc/autograd/python_foreach_functions.h"
#include "torch/csrc/autograd/python_variable.h"
#include "torch/csrc/autograd/utils/python_arg_parsing.h"
#include "torch/csrc/autograd/utils/wrap_outputs.h"
#include "torch/csrc/utils/device_lazy_init.h"
#include "torch/csrc/utils/out_types.h"
#include "torch/csrc/utils/pycfunction_helpers.h"
#include "torch/csrc/utils/python_arg_parser.h"
#include "torch/csrc/utils/structseq.h"

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
$ops_headers
#endif

using at::Backend;
using at::Device;
using at::DeviceGuard;
using at::Generator;
using at::IntArrayRef;
using at::Layout;
using at::OptionalDeviceGuard;
using at::Scalar;
using at::ScalarType;
using at::Tensor;
using at::TensorList;
using at::TensorOptions;

using torch::utils::check_out_type_matches;
using namespace torch::autograd::utils;

namespace torch::autograd {

${py_forwards}

static PyMethodDef foreach_functions[] = {
  ${py_method_defs}
  {NULL}
};

static PyObject* THPForeachVariableFunctionsModule = nullptr;

void initForeachFunctions(PyObject* module) {
  static struct PyModuleDef def = {
      PyModuleDef_HEAD_INIT,
      "torch.foreach",
      nullptr,
      -1,
      foreach_functions};
  PyObject* foreach = PyModule_Create(&def);
  THPForeachVariableFunctionsModule = foreach;
  if (!foreach) {
    throw python_error();
  }
  if (PyModule_AddObject(module, "_foreach", foreach) != 0) {
    throw python_error();
  }
}

${py_methods}

} // namespace torch::autograd
