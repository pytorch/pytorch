#if !defined(C10_MOBILE) && !defined(ANDROID)
#include <torch/csrc/inductor/aoti_runner/model_container_runner_cuda.h>

namespace torch::inductor {

AOTIModelContainerRunnerCuda::AOTIModelContainerRunnerCuda(
    const std::string& model_so_path,
    size_t num_models,
    const std::string& device_str,
    const std::string& cubin_dir)
    : AOTIModelContainerRunner(
          model_so_path,
          num_models,
          device_str,
          cubin_dir) {}

AOTIModelContainerRunnerCuda::~AOTIModelContainerRunnerCuda() {}

std::vector<at::Tensor> AOTIModelContainerRunnerCuda::run(
    std::vector<at::Tensor>& inputs) {
  at::cuda::CUDAStream cuda_stream = c10::cuda::getCurrentCUDAStream();
  return AOTIModelContainerRunner::run(
      inputs, reinterpret_cast<AOTInductorStreamHandle>(cuda_stream.stream()));
}

std::vector<at::Tensor> AOTIModelContainerRunnerCuda::run_with_cuda_stream(
    std::vector<at::Tensor>& inputs,
    at::cuda::CUDAStream cuda_stream) {
  return AOTIModelContainerRunner::run(
      inputs, reinterpret_cast<AOTInductorStreamHandle>(cuda_stream.stream()));
}

} // namespace torch::inductor
#endif
