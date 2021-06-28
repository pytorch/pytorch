#pragma once

#include <torch/csrc/python_headers.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <ATen/ATen.h>

namespace torch { namespace autograd {

struct TORCH_API SavedVariableHooks {
  virtual PyObject* call_pack_hook(at::Tensor &tensor) = 0;
  virtual at::Tensor call_unpack_hook(PyObject* obj) = 0 ;
};

}}
