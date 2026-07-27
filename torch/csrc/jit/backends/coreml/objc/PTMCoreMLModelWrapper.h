#include <ATen/core/ivalue.h>
#include <torch/csrc/jit/backends/coreml/objc/PTMCoreMLExecutor.h>
#include <torch/csrc/jit/backends/coreml/objc/PTMCoreMLTensorSpec.h>

namespace torch {
namespace jit {
namespace mobile {
namespace coreml {

class MLModelWrapper : public CustomClassHolder {
 public:
  PTMCoreMLExecutor* executor;
  std::vector<TensorSpec> outputs;

  MLModelWrapper() = delete;

  MLModelWrapper(PTMCoreMLExecutor* executor) : executor(executor) {
    [executor retain];
  }

  MLModelWrapper(const MLModelWrapper& oldObject) {
    executor = oldObject.executor;
    outputs = oldObject.outputs;
    [executor retain];
  }

  MLModelWrapper(MLModelWrapper&& oldObject) {
    executor = oldObject.executor;
    outputs = oldObject.outputs;
    [executor retain];
  }

  ~MLModelWrapper() {
    [executor release];
  }
};

} // namespace coreml
} // namespace mobile
} // namespace jit
} // namespace torch
