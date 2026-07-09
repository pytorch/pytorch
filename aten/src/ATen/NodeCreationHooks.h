#pragma once

#include <c10/core/SafePyObject.h>
#include <c10/macros/Export.h>
#include <optional>
#include <utility>
#include <vector>

namespace at::impl {

// TLS for torch.autograd.graph.node_creation_hook. Lives in ATen (rather than
// torch/csrc/autograd) so that at::ThreadLocalState can snapshot it across
// thread boundaries, e.g. into autograd engine worker threads. This mirrors
// SavedTensorDefaultHooksTLS.
struct TORCH_API NodeCreationHooksEntry {
  // Either hook may be nullopt. They are attached to every node created while
  // the context is live, as backward pre/post hooks (see
  // Node::register_prehook/register_hook).
  std::optional<c10::SafePyObject> prehook;
  std::optional<c10::SafePyObject> posthook;
  // If true, the posthook is also run when the node's backward raises (with
  // empty grad_inputs), so it can restore state. See register_forward_hook's
  // always_call.
  bool always_call = false;
};

struct TORCH_API NodeCreationHooksTLS {
  // Stored in registration order (outermost context manager first).
  std::vector<NodeCreationHooksEntry> stack;
};

struct TORCH_API NodeCreationHooks {
  static void push_hooks(
      std::optional<c10::SafePyObject> prehook,
      std::optional<c10::SafePyObject> posthook,
      bool always_call);
  static void pop_hooks();
  static bool empty();

  static const NodeCreationHooksTLS& get_tls_state();
  static void set_tls_state(const NodeCreationHooksTLS& state);
};

} // namespace at::impl
