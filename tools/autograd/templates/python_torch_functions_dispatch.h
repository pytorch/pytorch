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
using at::SparseTensorRef;
using at::Storage;

static at::Type& default_type() {
  return torch::tensor::get_default_tensor_type();
}

static void maybe_initialize_cuda(const at::Type &type) {
  if (type.is_cuda()) {
    torch::utils::cuda_lazy_init();
  }
}

// manual dispatch code for clamp
inline Tensor dispatch_clamp(const Tensor & self, Scalar min, Scalar max) {
  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.clamp(min, max);
}
inline Tensor dispatch_clamp_min(const Tensor & self, Scalar min) {
  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.clamp_min(min);
}
inline Tensor dispatch_clamp_max(const Tensor & self, Scalar max) {
  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.clamp_max(max);
}
inline Tensor & dispatch_clamp(const Tensor & self, Scalar min, Scalar max, Tensor result) {
  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::clamp_out(result, self, min, max);
}
inline Tensor & dispatch_clamp_min(const Tensor & self, Scalar min, Tensor result) {
  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::clamp_min_out(result, self, min);
}
inline Tensor & dispatch_clamp_max(const Tensor & self, Scalar max, Tensor result) {
  AutoNoGIL no_gil;
  AutoGPU auto_gpu(result);
  return at::clamp_max_out(result, self, max);
}

${py_method_dispatch}

}} // namespace torch::autograd
