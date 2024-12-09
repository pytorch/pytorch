#if !defined(C10_MOBILE) && !defined(ANDROID)
#include <torch/csrc/inductor/aoti_runner/model_container_runner_xpu.h>

namespace torch::inductor {

AOTIModelContainerRunnerXpu::AOTIModelContainerRunnerXpu(
    const std::string& model_so_path,
    size_t num_models,
    const std::string& device_str,
    const std::string& cubin_dir)
    : AOTIModelContainerRunner(
          model_so_path,
          num_models,
          device_str,
          cubin_dir) {}

AOTIModelContainerRunnerXpu::~AOTIModelContainerRunnerXpu() = default;

std::vector<at::Tensor> AOTIModelContainerRunnerXpu::run(
    std::vector<at::Tensor>& inputs) {
  at::xpu::XPUStream xpu_stream = c10::xpu::getCurrentXPUStream();
  return AOTIModelContainerRunner::run(
      inputs, reinterpret_cast<AOTInductorStreamHandle>(&(xpu_stream.queue())));
}

std::vector<at::Tensor> AOTIModelContainerRunnerXpu::run_with_xpu_stream(
    std::vector<at::Tensor>& inputs,
    at::xpu::XPUStream xpu_stream) {
  return AOTIModelContainerRunner::run(
      inputs, reinterpret_cast<AOTInductorStreamHandle>(&(xpu_stream.queue())));
}

} // namespace torch::inductor
#endif
