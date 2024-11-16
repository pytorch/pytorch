#include <ATen/SavedTensorHooks.h>
#include <c10/util/Exception.h>
#include <stack>
#include <utility>
#include <c10/core/SafePyObject.h>

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
  if (!tls.stack.empty()) {
    assertSavedTensorHooksNotDisabled();
  }
}

void SavedTensorDefaultHooks::enable() {
  tls.disabled_error_message = std::nullopt;
}

/* static */ bool SavedTensorDefaultHooks::set_tracing(bool is_tracing) {
  bool prior  = tls.is_tracing;
  tls.is_tracing = is_tracing;
  return prior;
}

const std::optional<std::string>& SavedTensorDefaultHooks::get_disabled_error_message() {
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

void SavedTensorDefaultHooks::push_hooks(SafePyObject pack_hook, SafePyObject unpack_hook) {
  TORCH_INTERNAL_ASSERT(is_initialized);
  assertSavedTensorHooksNotDisabled();
  tls.stack.emplace(std::move(pack_hook), std::move(unpack_hook));
}

std::pair<SafePyObject, SafePyObject> SavedTensorDefaultHooks::pop_hooks() {
  TORCH_INTERNAL_ASSERT(is_initialized && !tls.stack.empty());
  std::pair<SafePyObject, SafePyObject> hooks = std::move(tls.stack.top());
  tls.stack.pop();
  return hooks;
}

std::optional<std::pair<SafePyObject, SafePyObject>> SavedTensorDefaultHooks::get_hooks() {
  // For tls.is_tracing, see NOTE: [Deferring tensor pack/unpack hooks until runtime]
  if (!is_initialized || tls.stack.empty() || tls.is_tracing) {
    return std::nullopt;
  }
  return tls.stack.top();
}

}
