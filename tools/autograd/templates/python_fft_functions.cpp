// ${generated_comment}

#include "torch/csrc/Device.h"
#include "torch/csrc/DynamicTypes.h"
#include "torch/csrc/Exceptions.h"
#include "torch/csrc/autograd/python_fft_functions.h"
#include "torch/csrc/autograd/python_variable.h"
#include "torch/csrc/autograd/utils/wrap_outputs.h"
#include "torch/csrc/autograd/utils/python_arg_parsing.h"
#include "torch/csrc/autograd/generated/variable_factories.h"
#include "torch/csrc/utils/out_types.h"
#include "torch/csrc/utils/pycfunction_helpers.h"
#include "torch/csrc/utils/python_arg_parser.h"
#include "torch/csrc/utils/structseq.h"
#include "torch/csrc/utils/cuda_lazy_init.h"

#include <ATen/ATen.h>

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
