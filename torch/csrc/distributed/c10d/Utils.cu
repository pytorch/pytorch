#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/csrc/distributed/c10d/Utils.hpp>
#include <torch/torch.h>
#include <algorithm>

namespace c10d {

template <typename T>
__global__ void checkForNaN(T* data, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = tid; i < size; i += stride) {
    CUDA_KERNEL_ASSERT(!isnan(data[i]));
  }
}

void checkForNan(const at::Tensor& tensor) {
  // skip check for non float types
  if (!torch::is_floating_point(tensor)) {
    return;
  }
  const int64_t maxNumThreadsPerBlock = 512;
  const int64_t maxNumBlocks = 24;
  const int numThreadsPerBlock =
      std::min(maxNumThreadsPerBlock, tensor.numel());

  const int numBlocks = std::min(
      maxNumBlocks,
      (tensor.numel() + numThreadsPerBlock - 1) / numThreadsPerBlock);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(tensor.scalar_type(), "checkForNaN", [&] {
    checkForNaN<scalar_t><<<numBlocks, numThreadsPerBlock>>>(
        tensor.data_ptr<scalar_t>(), tensor.numel());
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace c10d
