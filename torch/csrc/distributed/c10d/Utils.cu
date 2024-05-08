#include <torch/csrc/distributed/c10d/Utils.hpp>

#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace c10d {

__global__ void checkForNaN(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = data[idx];
        // Use device-side assert to check for NaN
        // This will cause the kernel to terminate if NaN is found
        CUDA_KERNEL_ASSERT(!isnan(val));
    }
}

void checkForNan(const at::Tensor& tensor) {
  const int blockSize = 256;
  const int numBlocks = (tensor.numel() + blockSize - 1) / blockSize;

  checkForNaN<<<numBlocks, blockSize>>>(tensor.data_ptr<float>(), tensor.numel());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}
