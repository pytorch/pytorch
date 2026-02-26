#include <torch/csrc/inductor/aoti_runner/model_container_runner_cuda.h>
#include <torch/nativert/executor/AOTInductorDelegateExecutor.h>

namespace torch::nativert {

namespace {
std::unique_ptr<torch::inductor::AOTIModelContainerRunner>
create_aoti_model_container_runner_cuda(
    const std::string& model_so_path,
    size_t num_models,
    const std::string& device_str,
    const std::string& cubin_dir,
    const bool run_single_threaded) {
  return std::make_unique<torch::inductor::AOTIModelContainerRunnerCuda>(
      model_so_path, num_models, device_str, cubin_dir, run_single_threaded);
}
} // namespace

C10_REGISTER_TYPED_CREATOR(
    AOTIModelContainerRunnerRegistry,
    at::kCUDA,
    create_aoti_model_container_runner_cuda)

} // namespace torch::nativert
