#include <c10/core/impl/PyInterpreterTLS.h>

namespace c10 {
namespace impl {

// This is only ever populated by a non-torchdeploy Python interpreter
const PyInterpreter* globalPyInterpreterState = nullptr;

// This is only ever populated by torchdeploy Python interpreters
thread_local const PyInterpreter* pyInterpreterState = nullptr;

void PyInterpreterTLS::set_global_state(const PyInterpreter* state) {
  globalPyInterpreterState = state;
}

void PyInterpreterTLS::set_state(const PyInterpreter* state) {
  pyInterpreterState = state;
}

const PyInterpreter* PyInterpreterTLS::get_state() {
  if (pyInterpreterState) return pyInterpreterState;
  return globalPyInterpreterState;
}

void PyInterpreterTLS::reset_state() {
  pyInterpreterState = nullptr;
}

} // namespace impl
} // namespace c10
