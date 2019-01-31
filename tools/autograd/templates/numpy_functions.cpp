// ${generated_comment}

// Python bindings for torch.* functions implemented through ATen.
//
// The functions are bound as static methods on a class
// torch._C._VariableFunctions which is also aliased as Variable._torch
// and also copied into 'torch' module.

#include <Python.h>

#include "python_torch_functions_dispatch.h"

#include "torch/csrc/autograd/python_variable.h"
#include "torch/csrc/autograd/utils/wrap_outputs.h"
#include "torch/csrc/Dtype.h"
#include "torch/csrc/DynamicTypes.h"
#include "torch/csrc/Exceptions.h"
#include "torch/csrc/utils/python_arg_parser.h"
#include "torch/csrc/utils/tensor_layouts.h"
#include "torch/csrc/utils/tensor_new.h"
#include "torch/csrc/utils/tensor_numpy.h"
#include "torch/csrc/jit/tracer.h"
#include "torch/csrc/autograd/generated/variable_factories.h"
#include "torch/csrc/utils/structseq.h"

#include <ATen/ATen.h>

#include <functional>
#include <initializer_list>
#include <stdexcept>
#include <utility>

using at::Tensor;
using at::Device;
using at::Scalar;
using at::ScalarType;
using at::Backend;
using at::OptionalDeviceGuard;
using at::DeviceGuard;
using at::TensorOptions;

using namespace torch::autograd::utils;


namespace torch {
namespace autograd {
// generated methods start here

${py_methods}

PyMethodDef numpy_functions[] = {
  ${py_method_defs}
  {NULL}
};

}} // namespace torch::autograd
