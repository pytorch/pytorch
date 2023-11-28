#pragma once

#include <torch/csrc/inductor/aoti_model_container_runner.h>

namespace torch::inductor {

class TORCH_API AOTIModelContainerRunnerCuda : public AOTIModelContainerRunner {
 public:
  AOTIModelContainerRunnerCuda(
      const char* model_path,
      size_t num_models = 1,
      const char* cubin_dir = nullptr)
      : AOTIModelContainerRunner(model_path, num_models, false, cubin_dir) {}

  std::vector<at::Tensor> run(
      std::vector<at::Tensor> inputs,
      AOTInductorStreamHandle cuda_stream_handle = nullptr,
      AOTIProxyExecutorHandle proxy_executor_handle = nullptr);
};

} // namespace torch::inductor
