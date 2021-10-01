#include <ATen/core/PythonModeTLS.h>

namespace at { namespace impl {

thread_local std::vector<std::shared_ptr<TorchDispatchTypeObject>> pythonModeStack;

void PythonModeTLS::push_mode(const std::shared_ptr<TorchDispatchTypeObject>& state) {
  pythonModeStack.push_back(state);
  c10::impl::tls_set_dispatch_key_included(DispatchKey::PythonMode, true);
}

std::shared_ptr<TorchDispatchTypeObject> PythonModeTLS::pop_mode() {
  auto result = pythonModeStack.back();
  pythonModeStack.pop_back();
  if (pythonModeStack.size() == 0) {
    c10::impl::tls_set_dispatch_key_included(DispatchKey::PythonMode, false);
  }
  return result;
}

const std::vector<std::shared_ptr<TorchDispatchTypeObject>>& PythonModeTLS::get_state() {
  return pythonModeStack;
}

void PythonModeTLS::set_state(const std::vector<std::shared_ptr<TorchDispatchTypeObject>>& state) {
  pythonModeStack.clear();
  std::copy(state.begin(), state.end(), pythonModeStack.begin());
}

} // namespace impl
} // namespace at
