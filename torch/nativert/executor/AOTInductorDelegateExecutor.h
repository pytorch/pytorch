#pragma once

#include <torch/csrc/inductor/aoti_runner/model_container_runner.h>
#include <torch/csrc/inductor/aoti_runner/model_container_runner_cpu.h>
#include <torch/nativert/executor/ETDelegateExecutor.h>
#include <torch/nativert/executor/ExecutorConfig.h>

#ifdef USE_CUDA
#include <torch/csrc/inductor/aoti_runner/model_container_runner_cuda.h>
#endif

namespace torch::nativert {

class AOTIDelegateExecutor : public ETDelegateExecutor {
 public:
  explicit AOTIDelegateExecutor(
      const Node& node,
      const std::shared_ptr<Weights>& weights,
      const ExecutorConfig& executorConfig,
      caffe2::serialize::PyTorchStreamReader* packageReader,
      const MakeProxyExecutorFn& makeProxyExecutorFunc);
  ~AOTIDelegateExecutor() override = default;

  void processWeights(std::shared_ptr<Weights> weights) override;
  void initWeights(std::shared_ptr<Weights> weights) override;
  void commitWeights() override;

  std::vector<at::Tensor> run(std::vector<at::Tensor>& inputs) override;

 private:
  std::unique_ptr<torch::inductor::AOTIModelContainerRunner>
      aoti_model_container_runner_;

  // key is weight's original fqn, value is weight's name in AOTI
  std::unordered_map<std::string, std::string> weight_names_map_;
};

C10_DECLARE_TYPED_REGISTRY(
    AOTIModelContainerRunnerRegistry,
    c10::DeviceType,
    torch::inductor::AOTIModelContainerRunner,
    std::unique_ptr,
    const std::string&,
    size_t,
    const std::string&,
    const std::string&,
    const bool);

} // namespace torch::nativert
