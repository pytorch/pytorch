#if !defined(C10_MOBILE) && !defined(ANDROID)
#include <torch/csrc/inductor/aoti_runner/model_container_runner_xpu.h>

namespace torch::inductor {

AOTIModelContainerRunnerXpu::AOTIModelContainerRunnerXpu(
    const std::string& model_so_path,
    size_t num_models,
    const std::string& device_str,
    const std::string& kernel_bin_dir,
    const bool run_single_threaded)
    : AOTIModelContainerRunner(
          model_so_path,
          num_models,
          device_str,
          kernel_bin_dir,
          run_single_threaded) {}

AOTIModelContainerRunnerXpu::~AOTIModelContainerRunnerXpu() = default;

std::vector<at::Tensor> AOTIModelContainerRunnerXpu::run_impl(
    std::vector<AtenTensorHandle>& input_handles,
    void* stream_handle) {
  if (stream_handle == nullptr) {
    at::xpu::XPUStream xpu_stream = c10::xpu::getCurrentXPUStream();
    stream_handle = reinterpret_cast<void*>(&(xpu_stream.queue()));
  }
  return AOTIModelContainerRunner::run_impl(input_handles, stream_handle);
}

std::vector<at::Tensor> AOTIModelContainerRunnerXpu::run_with_xpu_stream(
    const std::vector<at::Tensor>& inputs,
    const at::xpu::XPUStream& xpu_stream) {
  return run(inputs, reinterpret_cast<void*>(&(xpu_stream.queue())));
}

namespace {
std::unique_ptr<AOTIModelContainerRunner> create_aoti_runner_xpu(
    const std::string& model_so_path,
    size_t num_models,
    const std::string& device_str,
    const std::string& kernel_bin_dir,
    const bool run_single_threaded) {
  return std::make_unique<AOTIModelContainerRunnerXpu>(
      model_so_path,
      num_models,
      device_str,
      kernel_bin_dir,
      run_single_threaded);
}
} // namespace

RegisterAOTIModelRunner register_xpu_runner("xpu", &create_aoti_runner_xpu);
} // namespace torch::inductor
#endif
