#pragma once

#include <cuda_runtime_api.h>
#include <torch/csrc/inductor/aoti_runner/model_container_runner.h>

namespace torch::inductor {

class TORCH_API AOTIModelContainerRunnerCuda : public AOTIModelContainerRunner {
 public:
  AOTIModelContainerRunnerCuda(
      const std::string& model_so_path,
      size_t num_models = 1,
      const std::string& cubin_dir = "")
      : AOTIModelContainerRunner(model_so_path, num_models, false, cubin_dir) {}

  std::vector<at::Tensor> run(
      std::vector<at::Tensor>& inputs,
      cudaStream_t cuda_stream_handle = nullptr);
};

} // namespace torch::inductor
