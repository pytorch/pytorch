#include <torch/csrc/autograd/engine.h>
#include <torch/csrc/dynamo/compiled_autograd.h>

namespace torch::dynamo::autograd {

static std::unique_ptr<PyCompilerInterface> kActivePyCompilerInterface;

const std::unique_ptr<PyCompilerInterface>& getPyCompilerInterface() {
  TORCH_INTERNAL_ASSERT(kActivePyCompilerInterface != nullptr);
  return kActivePyCompilerInterface;
}

PyCompilerGuard::PyCompilerGuard(std::unique_ptr<PyCompilerInterface>&& impl) {
  TORCH_INTERNAL_ASSERT(
      kActivePyCompilerInterface == nullptr && impl != nullptr);
  kActivePyCompilerInterface = std::move(impl);
}

PyCompilerGuard::~PyCompilerGuard() {
  TORCH_INTERNAL_ASSERT(kActivePyCompilerInterface != nullptr);
  kActivePyCompilerInterface.reset();
}

std::vector<std::optional<InputMetadata>> get_input_metadata(
    const edge_list& edges) {
  return torch::autograd::collect_input_metadata(edges);
}

} // namespace torch::dynamo::autograd
