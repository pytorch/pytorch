#pragma once

#include <c10/core/SafePyObject.h>
#include <c10/macros/Macros.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/Optional.h>

namespace c10 {
namespace impl {

struct C10_API TorchDispatchMode {
  static std::shared_ptr<SafePyObject> torchDispatchModeState;
  static void set_state(std::shared_ptr<SafePyObject> state);
  static std::shared_ptr<SafePyObject> get_state();
};

C10_API bool dispatch_mode_enabled();

} // namespace impl
} // namespace c10
