#if defined(__APPLE__)
#pragma once

#include <torch/csrc/inductor/aoti_runner/model_container_runner.h>

namespace torch::inductor {
class TORCH_API AOTIModelContainerRunnerMps : public AOTIModelContainerRunner {
 public:
  AOTIModelContainerRunnerMps(
      const std::string& model_so_path,
      size_t num_models = 1,
      const bool run_single_threaded = false);

  ~AOTIModelContainerRunnerMps() override;
};

} // namespace torch::inductor
#endif
