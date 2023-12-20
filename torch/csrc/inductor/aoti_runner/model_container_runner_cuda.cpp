#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/inductor/aoti_runner/model_container_runner_cuda.h>

namespace torch::inductor {

std::vector<at::Tensor> AOTIModelContainerRunnerCuda::run(
    std::vector<at::Tensor>& inputs,
    cudaStream_t cuda_stream_handle) {
  if (cuda_stream_handle == nullptr) {
    cuda_stream_handle = c10::cuda::getCurrentCUDAStream().stream();
  }
  return AOTIModelContainerRunner::run(
      inputs, reinterpret_cast<AOTInductorStreamHandle>(cuda_stream_handle));
}

} // namespace torch::inductor
