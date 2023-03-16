#include <ATen/core/ivalue.h>
#include <torch/csrc/jit/backends/coreml/objc/PTMCoreMLExecutor.h>
#include <torch/csrc/jit/backends/coreml/objc/PTMCoreMLTensorSpec.h>

namespace torch {
namespace jit {
namespace mobile {
namespace coreml {

class MLModelWrapper : public CustomClassHolder {
 public:
  std::string modelID;
  PTMCoreMLExecutor* executor;
  std::vector<TensorSpec> outputs;

  MLModelWrapper() = delete;

  MLModelWrapper(const std::string& modelID, PTMCoreMLExecutor* executor) : modelID(modelID), executor(executor) {
    [executor retain];
  }

  MLModelWrapper(const MLModelWrapper& oldObject) {
    modelID = oldObject.modelID;
    executor = oldObject.executor;
    outputs = oldObject.outputs;
    [executor retain];
  }

  MLModelWrapper(MLModelWrapper&& oldObject) {
    modelID = oldObject.modelID;
    executor = oldObject.executor;
    outputs = oldObject.outputs;
    [executor retain];
  }

  ~MLModelWrapper() {
    [executor release];
  }
};

}
}
}
}
