#if !defined(C10_MOBILE) && !defined(ANDROID)
#include <torch/csrc/inductor/aoti_runner/model_container_runner_cpu.h>

namespace torch::inductor {

// NOTICE: Following APIs are subject to change due to active development
// We provide NO BC guarantee for these APIs
AOTIModelContainerRunnerCpu::AOTIModelContainerRunnerCpu(
    const std::string& model_so_path,
    size_t num_models)
    : AOTIModelContainerRunner(model_so_path, num_models, "cpu", "") {}

AOTIModelContainerRunnerCpu::~AOTIModelContainerRunnerCpu() = default;

std::vector<at::Tensor> AOTIModelContainerRunnerCpu::run(
    const std::vector<at::Tensor>& inputs) {
  return AOTIModelContainerRunner::run_impl(inputs);
}

namespace {
std::unique_ptr<AOTIModelContainerRunner> create_aoti_runner_cpu(
    const std::string& model_so_path,
    size_t num_models,
    const std::string& device_str,
    const std::string& cubin_dir) {
  if (device_str != "cpu") {
    throw std::runtime_error("Incorrect device passed to aoti_runner_cpu");
  }
  return std::make_unique<AOTIModelContainerRunnerCpu>(
      model_so_path, num_models);
}
} // namespace

RegisterAOTIModelRunner register_cpu_runner("cpu", &create_aoti_runner_cpu);

} // namespace torch::inductor
#endif
