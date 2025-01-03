#include <torch/csrc/autograd/engine.h>
#include <torch/csrc/dynamo/compiled_autograd.h>

namespace torch::dynamo::autograd {

std::unique_ptr<PyCompilerInterface> kPyCompilerInterface;

const std::unique_ptr<PyCompilerInterface>& getPyCompilerInterface() {
  TORCH_INTERNAL_ASSERT(kPyCompilerInterface != nullptr);
  return kPyCompilerInterface;
}

void setPyCompilerInterface(std::unique_ptr<PyCompilerInterface>&& impl) {
  TORCH_INTERNAL_ASSERT(impl != nullptr);
  kPyCompilerInterface = std::move(impl);
}

void resetPyCompilerInterface() {
  kPyCompilerInterface.reset();
}

std::vector<c10::optional<InputMetadata>> get_input_metadata(
    const edge_list& edges) {
  return torch::autograd::collect_input_metadata(edges);
}

} // namespace torch::dynamo::autograd
