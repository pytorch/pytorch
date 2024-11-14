#if !defined(C10_MOBILE) && !defined(ANDROID)
#pragma once

#include <torch/csrc/inductor/aoti_runner/model_container_runner.h>

namespace torch::inductor {
class TORCH_API AOTIModelContainerRunnerCpu : public AOTIModelContainerRunner {
 public:
  AOTIModelContainerRunnerCpu(
      const std::string& model_so_path,
      size_t num_models = 1);

  ~AOTIModelContainerRunnerCpu() override;

  std::vector<at::Tensor> run(const std::vector<at::Tensor>& inputs) override;
};

} // namespace torch::inductor
#endif
