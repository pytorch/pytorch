#include <ATen/PythonTorchFunctionTLS.h>
#include <c10/core/TensorImpl.h>

namespace at::impl {

static thread_local PythonTorchFunctionTLS pythonTorchFunctionState;

void PythonTorchFunctionTLS::push_onto_stack(std::shared_ptr<SafePyObject> mode) {
  pythonTorchFunctionState.stack_.push_back(std::move(mode));
}

const std::shared_ptr<SafePyObject> PythonTorchFunctionTLS::pop_stack() {
  TORCH_CHECK(!pythonTorchFunctionState.stack_.empty(), "trying to pop from empty mode stack");
  auto out = pythonTorchFunctionState.stack_.back();
  pythonTorchFunctionState.stack_.pop_back();
  return out;
}

const std::shared_ptr<SafePyObject>& PythonTorchFunctionTLS::get_stack_at(int64_t idx) {
  TORCH_CHECK(idx < static_cast<int64_t>(pythonTorchFunctionState.stack_.size()), "Tried to get stack at idx that's too big");
  return pythonTorchFunctionState.stack_[idx];
}

int64_t PythonTorchFunctionTLS::stack_len() {
  return static_cast<int64_t>(pythonTorchFunctionState.stack_.size());
}

void PythonTorchFunctionTLS::set_disabled_state(TorchFunctionDisabledState disabled_state) {
  pythonTorchFunctionState.disabled_state_ = disabled_state;
}

TorchFunctionDisabledState PythonTorchFunctionTLS::get_disabled_state() {
  return pythonTorchFunctionState.disabled_state_;
}

void PythonTorchFunctionTLS::set_state(const PythonTorchFunctionTLS& state) {
  pythonTorchFunctionState = state;
}

const PythonTorchFunctionTLS& PythonTorchFunctionTLS::get_state() {
  return pythonTorchFunctionState;
}

bool torch_function_mode_enabled() {
  // Manually flatten because gcc is refusing to inline here.  Note
  // that we are still calling __tls_get_addr twice here with GCC,
  // presumably because of
  // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=81501 (which says
  // the fix ships in GCC 16), but forcing inlining still improves
  // performance.
  const auto& ptfs = pythonTorchFunctionState;
  return ptfs.disabled_state_ != TorchFunctionDisabledState::ALL_DISABLED && !ptfs.stack_.empty();
}

// This is needed to disambiguate the ternary torch function disabled states
bool torch_function_all_disabled() {
  return PythonTorchFunctionTLS::get_disabled_state() == TorchFunctionDisabledState::ALL_DISABLED;
}

} // namespace at::impl
