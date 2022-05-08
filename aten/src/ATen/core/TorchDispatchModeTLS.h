#pragma once

#include <c10/macros/Macros.h>
#include <torch/library.h>
#include <ATen/core/dispatch/Dispatcher.h>

namespace at {
namespace impl {

struct TORCH_API TorchDispatchModeTLS {
  static void set_state(std::shared_ptr<SafePyObject> state);
  static const std::shared_ptr<SafePyObject>& get_state();
  static void reset_state();
};

} // namespace impl
} // namespace at
