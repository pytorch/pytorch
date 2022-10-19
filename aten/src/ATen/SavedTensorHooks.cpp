#include <ATen/SavedTensorHooks.h>
#include <c10/util/Exception.h>
#include <stack>

namespace at {

namespace {
  thread_local impl::SavedTensorDefaultHooksTLS tls;

  // This flag is set to true the first time default hooks are registered
  // and left at true for the rest of the execution.
  // It's an optimization so that users who never use default hooks don't need to
  // read the thread_local variables pack_hook_ and unpack_hook_.
  static bool is_initialized(false);
}

static void assertSavedTensorHooksNotDisabled() {
  TORCH_CHECK(SavedTensorDefaultHooks::is_enabled(), tls.disabled_error_message.value());
}

bool SavedTensorDefaultHooks::is_enabled() {
  // See NOTE: [disabled_error_message invariant]
  return !tls.disabled_error_message.has_value();
}

void SavedTensorDefaultHooks::disable(const std::string& message) {
  tls.disabled_error_message = message;
  if (tls.stack.size() > 0) {
    assertSavedTensorHooksNotDisabled();
  }
}

void SavedTensorDefaultHooks::enable() {
  tls.disabled_error_message = c10::nullopt;
}

const c10::optional<std::string>& SavedTensorDefaultHooks::get_disabled_error_message() {
  return tls.disabled_error_message;
}

const impl::SavedTensorDefaultHooksTLS& SavedTensorDefaultHooks::get_tls_state() {
  return tls;
}

void SavedTensorDefaultHooks::set_tls_state(const impl::SavedTensorDefaultHooksTLS& state) {
  tls = state;
}

void SavedTensorDefaultHooks::lazy_initialize() {
  is_initialized = true;
}

void SavedTensorDefaultHooks::push_hooks(PyObject* pack_hook, PyObject* unpack_hook) {
  // Reference counting is handled by the caller of `push_hooks`
  TORCH_INTERNAL_ASSERT(is_initialized);
  TORCH_INTERNAL_ASSERT(pack_hook != nullptr && unpack_hook != nullptr);
  assertSavedTensorHooksNotDisabled();
  tls.stack.push(std::make_pair(pack_hook, unpack_hook));
}

void SavedTensorDefaultHooks::pop_hooks() {
  // Reference counting is handled by the caller of `pop_hooks`
  TORCH_INTERNAL_ASSERT(is_initialized && !tls.stack.empty());
  tls.stack.pop();
}

std::pair<PyObject*, PyObject*> SavedTensorDefaultHooks::get_hooks() {
  if (!is_initialized || tls.stack.empty()) {
    return std::make_pair(nullptr, nullptr);
  }
  return tls.stack.top();
}

std::stack<std::pair<PyObject*, PyObject*>> SavedTensorDefaultHooks::get_stack() {
  return tls.stack;
}

void SavedTensorDefaultHooks::set_stack(std::stack<std::pair<PyObject*, PyObject*>> stack_) {
  tls.stack = stack_;
}

}
