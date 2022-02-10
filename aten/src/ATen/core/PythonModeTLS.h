#pragma once

#include <c10/macros/Macros.h>
#include <torch/library.h>
#include <ATen/core/dispatch/Dispatcher.h>

namespace at {
namespace impl {

struct TORCH_API PythonModeTLS {
  static void set_state(const std::shared_ptr<TorchDispatchTypeObject>& state);
  static const std::shared_ptr<TorchDispatchTypeObject>& get_state();
  static void reset_state();
};

} // namespace impl
} // namespace at
