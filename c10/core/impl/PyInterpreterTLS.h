#pragma once

#include <c10/core/impl/PyInterpreter.h>
#include <c10/macros/Macros.h>

namespace c10 {
namespace impl {

// This TLS is set by torchdeploy ONLY and specifies what the
// current Python interpreter for a thread is.
struct C10_API PyInterpreterTLS {
  // For use by a non-torchdeploy python interpreter
  static void set_global_state(const PyInterpreter* state);
  // For use by a torchdeploy python interpreter
  static void set_state(const PyInterpreter* state);
  static const PyInterpreter* get_state();
  static void reset_state();
};

} // namespace impl
} // namespace c10
