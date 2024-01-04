#include <torch/csrc/inductor/aoti_runner/model_container_runner_cpu.h>

namespace torch::inductor {

AOTIModelContainerRunnerCpu::AOTIModelContainerRunnerCpu(
    const std::string& model_so_path,
    size_t num_models)
    : AOTIModelContainerRunner(model_so_path, num_models, true, "") {}

std::vector<at::Tensor> AOTIModelContainerRunnerCpu::run(
    std::vector<at::Tensor>& inputs) {
  return AOTIModelContainerRunner::run(inputs);
}

} // namespace torch::inductor
