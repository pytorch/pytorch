#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// ${generated_comment}

#include "torch/csrc/Device.h"
#include "torch/csrc/DynamicTypes.h"
#include "torch/csrc/Exceptions.h"
#include "torch/csrc/autograd/python_special_functions.h"
#include "torch/csrc/autograd/python_return_types.h"
#include "torch/csrc/autograd/python_variable.h"
#include "torch/csrc/autograd/utils/wrap_outputs.h"
#include "torch/csrc/autograd/utils/python_arg_parsing.h"
#include "torch/csrc/autograd/generated/variable_factories.h"
#include "torch/csrc/utils/out_types.h"
#include "torch/csrc/utils/pycfunction_helpers.h"
#include "torch/csrc/utils/python_arg_parser.h"
#include "torch/csrc/utils/structseq.h"
#include "torch/csrc/utils/cuda_lazy_init.h"

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
using at::Generator;
using at::TensorList;
using at::Dimname;
using at::DimnameList;

using torch::utils::check_out_type_matches;
using namespace torch::autograd::utils;

namespace torch { namespace autograd {

// generated forward declarations start here

${py_forwards}

static PyMethodDef special_functions[] = {
  ${py_method_defs}
  {NULL}
};

static PyObject* THPSpecialVariableFunctionsModule = NULL;

void initSpecialFunctions(PyObject* module) {
  static struct PyModuleDef def = {
     PyModuleDef_HEAD_INIT,
     "torch._C._special",
     NULL,
     -1,
     special_functions
  };
  PyObject* special = PyModule_Create(&def);
  THPSpecialVariableFunctionsModule = special;
  if (!special) {
    throw python_error();
  }
  // steals a reference to special
  if (PyModule_AddObject(module, "_special", special) != 0) {
    throw python_error();
  }
}

// generated methods start here

${py_methods}

}} // namespace torch::autograd
