#pragma once

#include <ATen/core/Tensor.h>
#include <c10/core/SafePyObject.h>

namespace torch::autograd {

struct TORCH_API SavedVariableHooks {
  SavedVariableHooks() = default;
  SavedVariableHooks(const SavedVariableHooks&) = delete;
  SavedVariableHooks& operator=(const SavedVariableHooks&) = delete;
  SavedVariableHooks(SavedVariableHooks&&) = delete;
  SavedVariableHooks& operator=(SavedVariableHooks&&) = delete;
  virtual void call_pack_hook(const at::Tensor& tensor) = 0;
  virtual at::Tensor call_unpack_hook() = 0;
  virtual ~SavedVariableHooks() = default;
  virtual std::optional<std::pair<c10::SafePyObject, c10::SafePyObject>>
  retrieve_unpack_hook_data() const {
    TORCH_CHECK(
        false, "Compiled Autograd only supports python saved tensor hooks ");
  }
  // Unlike retrieve_unpack_hook_data(), doesn't require pack data to exist.
  virtual std::optional<c10::SafePyObject> retrieve_unpack_hook() const {
    return std::nullopt;
  }
};

} // namespace torch::autograd
