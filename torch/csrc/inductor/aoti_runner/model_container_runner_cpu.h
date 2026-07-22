#if !defined(C10_MOBILE) && !defined(ANDROID)
#pragma once

#include <torch/csrc/inductor/aoti_runner/model_container_runner.h>

namespace torch::inductor {
class TORCH_API AOTIModelContainerRunnerCpu : public AOTIModelContainerRunner {
 public:
  AOTIModelContainerRunnerCpu(
      const std::string& model_so_path,
      size_t num_models = 1,
      const bool run_single_threaded = false);

  // Construct with externally-provided weights (zero CPU allocation for
  // constants — the .so weights are not loaded). Caller retains tensor
  // ownership; tensors must outlive the runner.
  AOTIModelContainerRunnerCpu(
      const std::string& model_so_path,
      size_t num_models,
      std::unordered_map<std::string, at::Tensor>& constants);

  ~AOTIModelContainerRunnerCpu() override;
};

} // namespace torch::inductor
#endif
