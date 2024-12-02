#include <torch/csrc/dynamo/compiled_autograd.h>

namespace torch::dynamo::autograd {

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
