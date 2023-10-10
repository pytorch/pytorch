#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/inductor/aoti_model_runner_cuda.h>

namespace torch::inductor {

std::vector<at::Tensor> AOTIModelRunnerCuda::run(
    std::vector<at::Tensor> inputs,
    AOTInductorStreamHandle cuda_stream_handle,
    AOTIProxyExecutorHandle proxy_executor_handle) {
  if (cuda_stream_handle == nullptr) {
    cudaStream_t stream_id = c10::cuda::getCurrentCUDAStream().stream();
    cuda_stream_handle = reinterpret_cast<AOTInductorStreamHandle>(stream_id);
  }
  return AOTIModelRunner::run(
      inputs, cuda_stream_handle, proxy_executor_handle);
}

} // namespace torch::inductor
