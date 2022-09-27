#pragma once

#include <c10/core/SafePyObject.h>
#include <c10/macros/Macros.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/Optional.h>

namespace c10 {
namespace impl {

struct C10_API TorchDispatchModeTLS {
  static void set_state(std::shared_ptr<SafePyObject> state);
  static const std::shared_ptr<SafePyObject>& get_state();
  static void reset_state();
};

C10_API bool dispatch_mode_enabled();

} // namespace impl
} // namespace c10
