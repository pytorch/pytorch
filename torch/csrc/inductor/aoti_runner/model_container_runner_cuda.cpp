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

AOTIModelContainerRunnerCuda::~AOTIModelContainerRunnerCuda() = default;

std::vector<at::Tensor> AOTIModelContainerRunnerCuda::run(
    const std::vector<at::Tensor>& inputs) {
  at::cuda::CUDAStream cuda_stream = c10::cuda::getCurrentCUDAStream();
  return AOTIModelContainerRunner::run(
      inputs, reinterpret_cast<AOTInductorStreamHandle>(cuda_stream.stream()));
}

std::vector<at::Tensor> AOTIModelContainerRunnerCuda::run_with_cuda_stream(
    const std::vector<at::Tensor>& inputs,
    at::cuda::CUDAStream cuda_stream) {
  return AOTIModelContainerRunner::run(
      inputs, reinterpret_cast<AOTInductorStreamHandle>(cuda_stream.stream()));
}

namespace {
std::unique_ptr<AOTIModelContainerRunner> create_aoti_runner_cuda(
    const std::string& model_so_path,
    size_t num_models,
    const std::string& device_str,
    const std::string& cubin_dir) {
  return std::make_unique<AOTIModelContainerRunnerCuda>(
      model_so_path, num_models, device_str, cubin_dir);
}
} // namespace

RegisterAOTIModelRunner register_cuda_runner("cuda", &create_aoti_runner_cuda);

} // namespace torch::inductor
#endif
