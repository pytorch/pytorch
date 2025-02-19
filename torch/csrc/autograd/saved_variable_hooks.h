#pragma once

#include <ATen/core/Tensor.h>
#include <c10/core/SafePyObject.h>

namespace torch::autograd {

struct TORCH_API SavedVariableHooks {
  virtual void call_pack_hook(const at::Tensor& tensor) = 0;
  virtual at::Tensor call_unpack_hook() = 0;
  virtual ~SavedVariableHooks() = default;
  virtual std::optional<std::pair<c10::SafePyObject, c10::SafePyObject>>
  get_hook_for_compiled_autograd() const {
    return {};
  }
};

} // namespace torch::autograd
