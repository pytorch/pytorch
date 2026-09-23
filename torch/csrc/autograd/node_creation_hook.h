#pragma once

#include <c10/util/intrusive_ptr.h>
#include <torch/csrc/Export.h>

namespace torch::autograd {

struct Node;

// Abstract callback for torch.autograd.graph.node_creation_hook. Lives in
// libtorch_cpu so fire_node_creation_hooks can invoke it directly; the Python
// implementation (PyNodeCreationHook) lives in libtorch_python. This mirrors
// SavedVariableHooks.
struct TORCH_API NodeCreationHook {
  virtual ~NodeCreationHook() = default;
  virtual void operator()(const c10::intrusive_ptr<Node>& node) = 0;
};

} // namespace torch::autograd
