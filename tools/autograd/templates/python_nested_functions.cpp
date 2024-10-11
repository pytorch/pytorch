#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// ${generated_comment}

#include "torch/csrc/Device.h"
#include "torch/csrc/DynamicTypes.h"
#include "torch/csrc/Exceptions.h"
#include "torch/csrc/autograd/python_nested_functions.h"
#include "torch/csrc/autograd/generated/python_return_types.h"
#include "torch/csrc/autograd/python_variable.h"
#include "torch/csrc/autograd/utils/wrap_outputs.h"
#include "torch/csrc/autograd/utils/python_arg_parsing.h"
#include "torch/csrc/autograd/generated/variable_factories.h"
#include "torch/csrc/utils/out_types.h"
#include "torch/csrc/utils/pycfunction_helpers.h"
#include "torch/csrc/utils/python_arg_parser.h"
#include "torch/csrc/utils/structseq.h"
#include "torch/csrc/utils/device_lazy_init.h"

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
$ops_headers
#endif

using at::Tensor;
using at::Device;
using at::Layout;
using at::Scalar;
using at::ScalarType;
using at::Backend;
using at::OptionalDeviceGuard;
using at::DeviceGuard;
using at::TensorOptions;
using at::IntArrayRef;
using at::OptionalIntArrayRef;
using at::Generator;
using at::TensorList;
using at::Dimname;
using at::DimnameList;

using namespace torch::autograd::utils;

namespace torch::autograd {

// generated forward declarations start here

${py_forwards}

static PyMethodDef nested_functions[] = {
  {NULL, NULL, 0, NULL},
  ${py_method_defs}
  {NULL}
};

static PyObject* THPNestedVariableFunctionsModule = NULL;

void initNestedFunctions(PyObject* module) {
  nested_functions[0] = get_nested_functions_manual()[0];
  static struct PyModuleDef def = {
     PyModuleDef_HEAD_INIT,
     "torch._C._nested",
     NULL,
     -1,
     nested_functions
  };
  PyObject* nested = PyModule_Create(&def);
  THPNestedVariableFunctionsModule = nested;
  if (!nested) {
    throw python_error();
  }
  // steals a reference to nested
  if (PyModule_AddObject(module, "_nested", nested) != 0) {
    throw python_error();
  }
}

// generated methods start here

${py_methods}

} // namespace torch::autograd
