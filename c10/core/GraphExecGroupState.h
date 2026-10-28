#pragma once

#include <c10/core/SafePyObject.h>
#include <c10/macros/Export.h>
#include <optional>

namespace c10 {

struct C10_API GraphExecGroupState {
  static const std::optional<SafePyObject>& get_tls_state();
  static void set_tls_state(const std::optional<SafePyObject>& state);
};

} // namespace c10
