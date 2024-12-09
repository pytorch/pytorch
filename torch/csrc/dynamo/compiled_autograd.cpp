#include <torch/csrc/dynamo/compiled_autograd.h>

namespace torch::dynamo::autograd {

thread_local bool kProxyNodesIntoGraphEnabled = true;

bool is_proxy_nodes_into_graph_enabled() {
  return kProxyNodesIntoGraphEnabled;
}

void set_proxy_nodes_into_graph_enabled(bool enabled) {
  kProxyNodesIntoGraphEnabled = enabled;
}

std::unique_ptr<PyCompilerInterface> kPyCompilerInterface;

const std::unique_ptr<PyCompilerInterface>& getPyCompilerInterface() {
  TORCH_INTERNAL_ASSERT(kPyCompilerInterface != nullptr);
  return kPyCompilerInterface;
}

void setPyCompilerInterface(std::unique_ptr<PyCompilerInterface>&& impl) {
  TORCH_INTERNAL_ASSERT(impl != nullptr);
  std::swap(kPyCompilerInterface, impl);
  TORCH_INTERNAL_ASSERT(kPyCompilerInterface != nullptr);
}

void resetPyCompilerInterface() {
  kPyCompilerInterface.reset();
}

} // namespace torch::dynamo::autograd
