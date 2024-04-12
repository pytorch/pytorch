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

AOTIEagerKernelRunnerCPU::AOTIEagerKernelRunnerCPU(
    const std::string& kernel_so_path)
    : AOTIEagerKernelRunner(kernel_so_path) {}

void AOTIEagerKernelRunnerCPU::operator()(
    AtenTensorHandle* input_handles,
    AtenTensorHandle* output_handles) {
  return AOTIEagerKernelRunner::operator()(input_handles, output_handles);
}

} // namespace torch::inductor
#endif
