#pragma once

#include <torch/csrc/inductor/aoti_runner/model_container_runner.h>

namespace torch::inductor {
class TORCH_API AOTIModelContainerRunnerCpu : public AOTIModelContainerRunner {
 public:
  AOTIModelContainerRunnerCpu(
      const std::string& model_so_path,
      size_t num_models = 1)
      : AOTIModelContainerRunner(model_so_path, num_models, true, "") {}

  std::vector<at::Tensor> run(std::vector<at::Tensor>& inputs) {
    return AOTIModelContainerRunner::run(inputs);
  }
};

} // namespace torch::inductor
