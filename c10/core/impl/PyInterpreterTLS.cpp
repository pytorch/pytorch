#include <c10/core/impl/PyInterpreterTLS.h>

namespace c10 {
namespace impl {

thread_local const PyInterpreter* pyInterpreterState;

void PyInterpreterTLS::set_state(const PyInterpreter* state) {
  pyInterpreterState = state;
}

const PyInterpreter* PyInterpreterTLS::get_state() {
  return pyInterpreterState;
}

void PyInterpreterTLS::reset_state() {
  pyInterpreterState = nullptr;
}

} // namespace impl
} // namespace c10
