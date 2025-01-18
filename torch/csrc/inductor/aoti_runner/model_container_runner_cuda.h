#if !defined(C10_MOBILE) && !defined(ANDROID)
#pragma once

#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/inductor/aoti_runner/model_container_runner.h>

namespace torch::inductor {

// NOTICE: Following APIs are subject to change due to active development
// We provide NO BC guarantee for these APIs
// NOLINTNEXTLINE(cppcoreguidelines-special-member-functions)
class TORCH_CUDA_CPP_API AOTIModelContainerRunnerCuda
    : public AOTIModelContainerRunner {
 public:
  // @param device_str: cuda device string, e.g. "cuda", "cuda:0"
  AOTIModelContainerRunnerCuda(
      const std::string& model_so_path,
      size_t num_models = 1,
      const std::string& device_str = "cuda",
      const std::string& cubin_dir = "");

  ~AOTIModelContainerRunnerCuda() override;

  std::vector<at::Tensor> run_impl(
      std::vector<AtenTensorHandle>& input_handles,
      void* stream_handle) override;

  std::vector<at::Tensor> run_with_cuda_stream(
      const std::vector<at::Tensor>& inputs,
      const at::cuda::CUDAStream& cuda_stream);
};

} // namespace torch::inductor
#endif
