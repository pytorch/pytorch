#pragma once

// ${generated_comment}

#include "torch/csrc/autograd/generated/VariableType.h"
#include "torch/csrc/autograd/generated/variable_factories.h"
#include "torch/csrc/tensor/python_tensor.h"
#include "torch/csrc/utils/cuda_lazy_init.h"

#include <ATen/ATen.h>
#include <pybind11/pybind11.h>

// Contains inline wrappers around ATen functions that release the GIL and
// switch to the correct CUDA device.

namespace torch { namespace autograd {

using at::Tensor;
using at::Scalar;
using at::ScalarType;
using at::TensorList;
using at::IntArrayRef;
using at::Generator;
using at::Storage;
using at::Layout;

${py_method_dispatch}

}} // namespace torch::autograd
