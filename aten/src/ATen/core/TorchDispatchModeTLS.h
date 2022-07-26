#pragma once

#include <c10/macros/Macros.h>
#include <torch/library.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/Optional.h>

namespace at {
namespace impl {

struct TORCH_API TorchDispatchModeTLS {
  static void set_state(std::shared_ptr<SafePyObject> state);
  static const std::shared_ptr<SafePyObject>& get_state();
  static void reset_state();
};

bool dispatch_mode_enabled();
bool tensor_has_dispatch(const at::Tensor& t);
bool tensorlist_has_dispatch(const c10::List<c10::optional<at::Tensor>>& li);
bool tensorlist_has_dispatch(at::ITensorListRef li);


} // namespace impl
} // namespace at
