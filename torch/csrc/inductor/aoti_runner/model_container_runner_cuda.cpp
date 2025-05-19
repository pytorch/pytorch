#if !defined(C10_MOBILE) && !defined(ANDROID)
#include <torch/csrc/inductor/aoti_runner/model_container_runner_cuda.h>

namespace {
void deleter(void* ptr) {
  // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
  cudaFree(ptr);
}
} // namespace

namespace torch::inductor {

AOTIModelContainerRunnerCuda::AOTIModelContainerRunnerCuda(
    const std::string& model_so_path,
    size_t num_models,
    const std::string& device_str,
    const std::string& cubin_dir,
    const bool run_single_threaded)
    : AOTIModelContainerRunner(
          model_so_path,
          num_models,
          device_str,
          cubin_dir,
          run_single_threaded) {}

AOTIModelContainerRunnerCuda::~AOTIModelContainerRunnerCuda() = default;

std::vector<at::Tensor> AOTIModelContainerRunnerCuda::run_impl(
    std::vector<AtenTensorHandle>& input_handles,
    void* stream_handle) {
  if (stream_handle == nullptr) {
    at::cuda::CUDAStream cuda_stream = c10::cuda::getCurrentCUDAStream();
    stream_handle = reinterpret_cast<void*>(cuda_stream.stream());
  }
  return AOTIModelContainerRunner::run_impl(input_handles, stream_handle);
}

std::vector<at::Tensor> AOTIModelContainerRunnerCuda::run_with_cuda_stream(
    const std::vector<at::Tensor>& inputs,
    const at::cuda::CUDAStream& cuda_stream) {
  return run(inputs, reinterpret_cast<void*>(cuda_stream.stream()));
}

std::vector<at::Tensor> AOTIModelContainerRunnerCuda::slim_tensor_run(
    std::vector<at::Tensor>&& inputs,
    void* stream_handle) {
  if (stream_handle == nullptr) {
    at::cuda::CUDAStream cuda_stream = c10::cuda::getCurrentCUDAStream();
    stream_handle = reinterpret_cast<void*>(cuda_stream.stream());
  }
  return slim_tensor_run_impl(std::move(inputs), stream_handle, deleter);
}

namespace {
std::unique_ptr<AOTIModelContainerRunner> create_aoti_runner_cuda(
    const std::string& model_so_path,
    size_t num_models,
    const std::string& device_str,
    const std::string& cubin_dir,
    const bool run_single_threaded) {
  return std::make_unique<AOTIModelContainerRunnerCuda>(
      model_so_path, num_models, device_str, cubin_dir, run_single_threaded);
}
} // namespace

static RegisterAOTIModelRunner register_cuda_runner(
    "cuda",
    &create_aoti_runner_cuda);

} // namespace torch::inductor
#endif
