#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/csrc/distributed/c10d/Utils.hpp>
#include <torch/torch.h>
#include <algorithm>

namespace c10d {

// CUDA kernel to check if data has NAN, device side assert
// is raised if NAN is found
template <typename T>
__global__ void checkForNaN(T* data, size_t size) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;

  for (size_t i = tid; i < size; i += stride) {
    CUDA_KERNEL_ASSERT(!isnan(data[i]));
  }
}

// CHECK if a Tensor contains NAN in any of its element
void checkForNan(const at::Tensor& tensor) {
  // skip check for non float types
  if (!torch::is_floating_point(tensor)) {
    return;
  }
  const size_t maxNumThreadsPerBlock = 256;
  const size_t maxNumBlocks = 24;
  const size_t numThreadsPerBlock =
      std::min<size_t>(maxNumThreadsPerBlock, tensor.numel());

  const size_t numBlocks = std::min<size_t>(
      maxNumBlocks,
      (tensor.numel() + numThreadsPerBlock - 1) / numThreadsPerBlock);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(tensor.scalar_type(), "checkForNaN", [&] {
    checkForNaN<scalar_t><<<numBlocks, numThreadsPerBlock>>>(
        tensor.data_ptr<scalar_t>(), tensor.numel());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });

}

} // namespace c10d
