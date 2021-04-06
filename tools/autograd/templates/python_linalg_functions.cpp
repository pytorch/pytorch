// ${generated_comment}

#include "torch/csrc/Device.h"
#include "torch/csrc/DynamicTypes.h"
#include "torch/csrc/Exceptions.h"
#include "torch/csrc/autograd/python_linalg_functions.h"
#include "torch/csrc/autograd/python_variable.h"
#include "torch/csrc/autograd/utils/wrap_outputs.h"
#include "torch/csrc/autograd/utils/python_arg_parsing.h"
#include "torch/csrc/utils/pycfunction_helpers.h"
#include "torch/csrc/utils/python_arg_parser.h"
#include "torch/csrc/utils/structseq.h"

using at::Tensor;
using at::Scalar;
using at::ScalarType;
using at::MemoryFormat;
using at::Generator;
using at::IntArrayRef;

using namespace torch::autograd::utils;

namespace torch { namespace autograd {

// generated forward declarations start here

${py_forwards}

static PyMethodDef linalg_functions[] = {
  ${py_method_defs}
  {NULL}
};

static PyObject* THPLinalgVariableFunctionsModule = NULL;

void initLinalgFunctions(PyObject* module) {
  static struct PyModuleDef def = {
     PyModuleDef_HEAD_INIT,
     "torch._C._linalg",
     NULL,
     -1,
     linalg_functions
  };
  PyObject* linalg = PyModule_Create(&def);
  THPLinalgVariableFunctionsModule = linalg;
  if (!linalg) {
    throw python_error();
  }
  // steals a reference to linalg
  if (PyModule_AddObject(module, "_linalg", linalg) != 0) {
    throw python_error();
  }
}

// generated methods start here

${py_methods}

}} // namespace torch::autograd
