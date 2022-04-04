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

void PythonTorchFunctionTLS::set_state(const PythonTorchFunctionTLS& state) {
  pythonTorchFunctionState = state;
}

const PythonTorchFunctionTLS& PythonTorchFunctionTLS::get_state() {
  return pythonTorchFunctionState;
}

} // namespace impl
} // namespace at
