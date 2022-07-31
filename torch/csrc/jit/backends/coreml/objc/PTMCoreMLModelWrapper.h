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
  int32_t load_id = 0;
  int32_t inferences = 0;
  size_t mem_limit = 0;

  MLModelWrapper() = delete;

  MLModelWrapper(PTMCoreMLExecutor* executor) : executor(executor) {
    [executor retain];
  }

  MLModelWrapper(const MLModelWrapper& oldObject) {
    executor = oldObject.executor;
    outputs = oldObject.outputs;
    load_id = oldObject.load_id;
    inferences = oldObject.inferences;
    mem_limit = oldObject.mem_limit;
    [executor retain];
  }

  MLModelWrapper(MLModelWrapper&& oldObject) {
    executor = oldObject.executor;
    outputs = oldObject.outputs;
    load_id = oldObject.load_id;
    inferences = oldObject.inferences;
    mem_limit = oldObject.mem_limit;
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
