#pragma once

#include <ATen/ATen.h>

namespace torch { namespace autograd {

struct TORCH_API SavedVariableHooks {
  virtual void call_pack_hook(const at::Tensor &tensor) = 0;
  virtual at::Tensor call_unpack_hook() = 0;
  virtual ~SavedVariableHooks() = default;
};

struct TORCH_API DefaultSavedVariableHooks {
  static std::unique_ptr<SavedVariableHooks> get_hooks() {
    return nullptr;
  }
};

}}
