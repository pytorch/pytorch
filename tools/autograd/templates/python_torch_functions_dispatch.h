#pragma once

// ${generated_comment}

#include <ATen/ATen.h>
#include "torch/csrc/utils/auto_gil.h"
#include "torch/csrc/utils/auto_gpu.h"
#include "torch/csrc/utils/cuda_lazy_init.h"
#include "torch/csrc/autograd/generated/VariableType.h"
#include "torch/csrc/tensor/python_tensor.h"

// Contains inline wrappers around ATen functions that release the GIL and
// switch to the correct CUDA device.

namespace torch { namespace autograd {

using at::Tensor;
using at::Scalar;
using at::TensorList;
using at::IntList;
using at::Generator;
using at::SparseTensor;
using at::Storage;

static at::Type& default_type() {
  return torch::tensor::get_default_tensor_type();
}

static void maybe_initialize_cuda(const at::Type &type) {
  if (type.is_cuda()) {
    torch::utils::cuda_lazy_init();
  }
}

${py_method_dispatch}

}} // namespace torch::autograd
