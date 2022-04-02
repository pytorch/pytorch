#include <ATen/PythonTorchFunctionTLS.h>

namespace at {
namespace impl {

static thread_local PythonTorchFunctionTLS pythonTorchFunctionState;

void PythonTorchFunctionTLS::set_disabled(bool disabled) {
  pythonTorchFunctionState.disabled_ = disabled;
}

bool PythonTorchFunctionTLS::is_disabled() {
  return pythonTorchFunctionState.disabled_;
}

bool PythonTorchFunctionTLS::exchange_skip_next(bool new_skip_next) {
  return std::exchange(
    pythonTorchFunctionState.skip_next_,
    new_skip_next);
}

bool PythonTorchFunctionTLS::peek_skip_next() {
  return pythonTorchFunctionState.skip_next_;
}

void PythonTorchFunctionTLS::set_state(const PythonTorchFunctionTLS& state) {
  pythonTorchFunctionState = state;
}

const PythonTorchFunctionTLS& PythonTorchFunctionTLS::get_state() {
  return pythonTorchFunctionState;
}

} // namespace impl
} // namespace at
