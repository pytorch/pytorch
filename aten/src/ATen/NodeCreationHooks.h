#pragma once

#include <c10/core/SafePyObject.h>
#include <c10/macros/Export.h>
#include <vector>

namespace at::impl {

// TLS for torch.autograd.graph.node_creation_hook. Lives in ATen (rather than
// torch/csrc/autograd) so that at::ThreadLocalState can snapshot it across
// thread boundaries, e.g. into autograd engine worker threads. This mirrors
// SavedTensorDefaultHooksTLS.
struct TORCH_API NodeCreationHooksTLS {
  // Hooks fire in registration order (outermost context manager first).
  std::vector<c10::SafePyObject> stack;

  // True while hooks are running; creating a new autograd node from inside a
  // hook is an error (it would otherwise recurse unboundedly).
  bool is_firing = false;
};

struct TORCH_API NodeCreationHooks {
  static void push_hook(c10::SafePyObject hook);
  static void pop_hook();
  static bool empty();
  // Returns the previous value.
  static bool set_is_firing(bool is_firing);

  static const NodeCreationHooksTLS& get_tls_state();
  static void set_tls_state(const NodeCreationHooksTLS& state);
};

} // namespace at::impl
