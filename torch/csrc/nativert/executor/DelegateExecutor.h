#pragma once

#include <memory>
#include <vector>

#include <torch/script.h>

namespace torch::nativert {

class Weights;

std::string extractToTemporaryFolder(
    std::shared_ptr<caffe2::serialize::PyTorchStreamReader> packageReader,
    const std::string& targetPath);

// This is the extension point for delegation backends.
// Please refer to AOTIDelegateExecutor as an example.
class DelegateExecutor {
 public:
  virtual ~DelegateExecutor() {}

  // Runtime calls processWeights() to pass the weights to the delegate backend.
  // Typically, a backend would perform some form of validation and processing,
  // such as constant folding. The processed weights stays in the inactivate
  // state until commitWeights() is called.
  //
  // Weights tensors are co-owned by the runtime and the delegate backend.
  // In the regular inference run() path, neither Runtime or Delegate backend
  // can modify the weights tensor.
  // To support inplace weight update, weight tensors are be exposed by
  // ModelRunner::getWeights() to an external caller. The external caller can
  // then modify the weight tensors in-place. Such mutation would instantly
  // affect the weight tensors in the delegate backend.
  // When a weight tensor is no longer used by the delegate backend, the backend
  // must release it by decreasing a refcount. Runtime would
  // also release the refcount for weight tensor if it's no longer activte. The
  // underlying storage for weight tensors will be freed when the refcount
  // reaches 0.
  virtual void processWeights(std::shared_ptr<Weights> weights) = 0;

  // This call activate the processed weights.
  virtual void commitWeights() = 0;

  virtual std::vector<at::Tensor> run(std::vector<at::Tensor>& inputs) = 0;
};

} // namespace torch::nativert
