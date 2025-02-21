#pragma once

#include <ATen/core/Tensor.h>
#include <c10/core/SafePyObject.h>

namespace torch::autograd {

// struct SavedVariableUnpackHookData {
//   c10::SafePyObject hook;
//   c10::SafePyObject data;
//   at::Tensor input;
// };

struct TORCH_API SavedVariableHooks {
  virtual void call_pack_hook(const at::Tensor& tensor) = 0;
  virtual at::Tensor call_unpack_hook() = 0;
  virtual ~SavedVariableHooks() = default;
  virtual std::optional<std::pair<c10::SafePyObject, c10::SafePyObject>>
  retrieve_unpack_hook_data() const {
    return std::nullopt;
  };
};

} // namespace torch::autograd
