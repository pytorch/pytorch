#pragma once

#include <c10/core/SafePyObject.h>
#include <c10/macros/Macros.h>
#include <c10/util/Optional.h>

namespace c10 {
namespace impl {

struct C10_API PythonDispatcherTLS {
  static void set_state(SafePyHandle state);
  static SafePyHandle get_state();
  static void reset_state();
};

} // namespace impl
} // namespace c10
