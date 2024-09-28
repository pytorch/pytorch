#pragma once

#include <ATen/core/Tensor.h>

namespace torch::autograd {

struct TORCH_API SavedVariableHooks {
  virtual void call_pack_hook(const at::Tensor& tensor) = 0;
  virtual at::Tensor call_unpack_hook() = 0;
  virtual ~SavedVariableHooks() = default;
};

} // namespace torch::autograd
