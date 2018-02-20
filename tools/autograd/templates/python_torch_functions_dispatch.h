#pragma once

// ${generated_comment}

#include <ATen/ATen.h>
#include "torch/csrc/utils/auto_gil.h"
#include "torch/csrc/utils/auto_gpu.h"
#include "torch/csrc/autograd/generated/VariableType.h"

// Contains inline wrappers around ATen functions that release the GIL and
// switch to the correct CUDA device.

extern at::Type* THPDefaultATenType;

namespace torch { namespace autograd {

using at::Tensor;
using at::Scalar;
using at::TensorList;
using at::IntList;
using at::Generator;
using at::SparseTensor;
using at::Storage;

static at::Type& default_type() {
  if (!THPDefaultATenType) {
    throw std::runtime_error("THPDefaultATenType not initialized");
  }
  return *VariableType::getType(*THPDefaultATenType);
}

static void lazy_init_cuda() {
  static std::once_flag once;
  std::call_once(once, []() {
    auto module = THPObjectPtr(PyImport_ImportModule("torch.cuda"));
    if (!module) throw python_error();
    auto res = THPObjectPtr(PyObject_CallMethod(module.get(), "_lazy_init", ""));
    if (!res) throw python_error();
  });
}

static void maybe_initialize_cuda(const at::Type &type) {
  if (type.is_cuda()) {
    lazy_init_cuda();
  }
}

${py_method_dispatch}

}} // namespace torch::autograd
