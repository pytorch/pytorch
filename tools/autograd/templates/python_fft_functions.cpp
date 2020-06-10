// ${generated_comment}

#include "torch/csrc/Device.h"
#include "torch/csrc/DynamicTypes.h"
#include "torch/csrc/Exceptions.h"
#include "torch/csrc/autograd/python_fft_functions.h"
#include "torch/csrc/autograd/python_variable.h"
#include "torch/csrc/autograd/utils/wrap_outputs.h"
#include "torch/csrc/autograd/utils/python_arg_parsing.h"
#include "torch/csrc/utils/python_arg_parser.h"
#include "torch/csrc/utils/structseq.h"

using at::Tensor;
using at::Scalar;
using at::MemoryFormat;
using at::Generator;
using at::IntArrayRef;

using namespace torch::autograd::utils;

namespace torch { namespace autograd {

// generated forward declarations start here

${py_forwards}

static PyMethodDef fft_functions[] = {
  ${py_method_defs}
  {NULL}
};

static PyObject* THPFFTVariableFunctionsModule = NULL;

void initFFTFunctions(PyObject* module) {
  static struct PyModuleDef def = {
     PyModuleDef_HEAD_INIT,
     "torch._C._fft",
     NULL,
     -1,
     fft_functions
  };
  PyObject* fft = PyModule_Create(&def);
  THPFFTVariableFunctionsModule = fft;
  if (!fft) {
    throw python_error();
  }
  // steals a reference to fft
  if (PyModule_AddObject(module, "_fft", fft) != 0) {
    throw python_error();
  }
}

// generated methods start here

${py_methods}

}} // namespace torch::autograd
