#pragma once

#include <torch/csrc/inductor/aoti_runner/model_container_runner.h>

namespace torch::inductor {
class TORCH_API AOTIModelContainerRunnerCpu : public AOTIModelContainerRunner {
 public:
  AOTIModelContainerRunnerCpu(const char* model_path, size_t num_models = 1)
      : AOTIModelContainerRunner(model_path, num_models, true, nullptr) {}

  std::vector<at::Tensor> run(
      std::vector<at::Tensor> inputs,
      AOTIProxyExecutorHandle proxy_executor_handle = nullptr) {
    return AOTIModelContainerRunner::run(
        inputs, nullptr, proxy_executor_handle);
  }

  std::vector<const char*> get_call_spec() {
    return AOTIModelContainerRunner::get_call_spec();
  }
};

} // namespace torch::inductor
