#include <ATen/NodeCreationHooks.h>
#include <c10/util/Exception.h>
#include <utility>

namespace at::impl {

namespace {
thread_local NodeCreationHooksTLS tls;
}

void NodeCreationHooks::push_hook(c10::SafePyObject hook) {
  tls.stack.emplace_back(std::move(hook));
}

void NodeCreationHooks::pop_hook() {
  TORCH_INTERNAL_ASSERT(!tls.stack.empty());
  tls.stack.pop_back();
}

bool NodeCreationHooks::empty() {
  return tls.stack.empty();
}

bool NodeCreationHooks::set_is_firing(bool is_firing) {
  return std::exchange(tls.is_firing, is_firing);
}

const NodeCreationHooksTLS& NodeCreationHooks::get_tls_state() {
  return tls;
}

void NodeCreationHooks::set_tls_state(const NodeCreationHooksTLS& state) {
  tls = state;
}

} // namespace at::impl
