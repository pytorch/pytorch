#pragma once

#include "torch/csrc/nativert/executor/DelegateExecutor.h"
#include "torch/csrc/nativert/executor/ExecutorConfig.h"

#include "torch/csrc/nativert/executor/AOTInductorModelImpl.h" // @manual=//sigmoid/core/executor:aoti_model_impl

namespace torch::nativert {

class Weights;
class Node;

class AOTIDelegateExecutor : public DelegateExecutor {
 public:
  explicit AOTIDelegateExecutor(
      const std::string& path,
      std::shared_ptr<Weights> weights,
      c10::Device device,
      const ExecutorConfig& executorConfig,
      std::shared_ptr<caffe2::serialize::PyTorchStreamReader> packageReader);
  ~AOTIDelegateExecutor() override {}

  void processWeights(std::shared_ptr<Weights> weights) override;

  void commitWeights() override;

  std::vector<at::Tensor> run(std::vector<at::Tensor>& inputs) override;

 private:
  std::unique_ptr<torch::aot_inductor::AOTInductorModelImpl>
      aotInductorModelImpl_;

  // key is weight's original fqn, value is weight's name in AOTI
  std::unordered_map<std::string, std::string> weightsNameMap_;
};

} // namespace torch::nativert
