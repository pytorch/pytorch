#if !defined(C10_MOBILE) && !defined(ANDROID)
#include <torch/csrc/inductor/aoti_runner/model_container_runner_cpu.h>

namespace torch::inductor {

// NOTICE: Following APIs are subject to change due to active development
// We provide NO BC guarantee for these APIs
AOTIModelContainerRunnerCpu::AOTIModelContainerRunnerCpu(
    const std::string& model_so_path,
    size_t num_models)
    : AOTIModelContainerRunner(model_so_path, num_models, "cpu", "") {}

AOTIModelContainerRunnerCpu::~AOTIModelContainerRunnerCpu() {}

std::vector<at::Tensor> AOTIModelContainerRunnerCpu::run(
    std::vector<at::Tensor>& inputs) {
  return AOTIModelContainerRunner::run(inputs);
}

} // namespace torch::inductor
#endif
