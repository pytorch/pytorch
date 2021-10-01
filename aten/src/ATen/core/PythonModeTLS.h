#pragma once

#include <c10/macros/Macros.h>
#include <torch/library.h>
#include <ATen/core/dispatch/Dispatcher.h>

namespace at {
namespace impl {

struct TORCH_API PythonModeTLS {
  static void push_mode(const std::shared_ptr<TorchDispatchTypeObject>& state);
  static std::shared_ptr<TorchDispatchTypeObject> pop_mode();

  static const std::vector<std::shared_ptr<TorchDispatchTypeObject>>& get_state();
  static void set_state(const std::vector<std::shared_ptr<TorchDispatchTypeObject>>&);
};

} // namespace impl
} // namespace at
