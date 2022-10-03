#pragma once

#include <c10/core/impl/PyInterpreter.h>
#include <c10/macros/Macros.h>

namespace c10 {
namespace impl {

// This TLS is set by torchdeploy ONLY and specifies what the
// current Python interpreter for a thread is.
//
// TODO: maybe we should also set this TLS for non-torchdeploy
// Python interpreters too; but in that case, it's better as
// a global variable
struct C10_API PyInterpreterTLS {
  static void set_state(const PyInterpreter* state);
  static const PyInterpreter* get_state();
  static void reset_state();
};

} // namespace impl
} // namespace c10
