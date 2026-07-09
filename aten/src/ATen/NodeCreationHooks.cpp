#include <ATen/NodeCreationHooks.h>
#include <c10/util/Exception.h>
#include <utility>

namespace at::impl {

namespace {
thread_local NodeCreationHooksTLS tls;

// Set true the first time any hook is registered and left true thereafter, so
// the common case of never using node_creation_hook avoids the thread_local
// read on the hot path (fire_node_creation_hooks runs on every grad_fn
// attachment). Mirrors SavedTensorDefaultHooks::is_initialized.
bool is_initialized(false);
} // namespace

void NodeCreationHooks::push_hooks(
    std::optional<c10::SafePyObject> prehook,
    std::optional<c10::SafePyObject> posthook,
    bool always_call) {
  is_initialized = true;
  tls.stack.push_back(
      {std::move(prehook), std::move(posthook), always_call});
}

void NodeCreationHooks::pop_hooks() {
  TORCH_INTERNAL_ASSERT(!tls.stack.empty());
  tls.stack.pop_back();
}

bool NodeCreationHooks::empty() {
  return !is_initialized || tls.stack.empty();
}

const NodeCreationHooksTLS& NodeCreationHooks::get_tls_state() {
  return tls;
}

void NodeCreationHooks::set_tls_state(const NodeCreationHooksTLS& state) {
  tls = state;
}

} // namespace at::impl
