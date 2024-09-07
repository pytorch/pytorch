#ifdef USE_C10D_NCCL

#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/torch.h>
#include <algorithm>
#include <torch/csrc/distributed/c10d/NanCheck.hpp>

namespace c10d {

// CUDA kernel to check if data has NAN, device side assert
// is raised if NAN is found
/*
template <typename T>
__global__ void checkForNaN(T* data, size_t size) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;

  for (size_t i = tid; i < size; i += stride) {
    CUDA_KERNEL_ASSERT(!isnan(data[i]));
  }
}
*/

union BytePack16 {
  ulong2 data;
  double f64[2];
};

typedef union BytePack16 BytePack;
#define UNROLL 8

template <typename T>
__device__ __forceinline__ void checkBytePack(BytePack* tmp) {
  constexpr int nT = sizeof(BytePack) / sizeof(T);
  T* data = (T*)tmp;
  #pragma unroll
  for (int i = 0; i < nT; i++) {
    if (isnan(data[i])) __trap();
  }
}

template <typename T>
__device__ __forceinline__ void checkChunk(BytePack* ptr) {
  BytePack tmp[UNROLL];
  int nWorkers = blockDim.x * gridDim.x;
  #pragma unroll 8
  for (int j = 0; j < UNROLL; j++) {
    tmp[j] = ptr[nWorkers * j];
  }
  #pragma unroll 8
  for (int j = 0; j < UNROLL; j++) {
    // if (isnan(tmp[j].f64[0]) || isnan(tmp[j].f64[1])) __trap();
    checkBytePack<T>(tmp + j);
  }
}

template <typename T>
__global__ void checkForNaN(T* data, size_t size) {
  BytePack* ptr = (BytePack*)data;
  size_t sizeInBP = size *  sizeof(T) / sizeof(BytePack);
  size_t loopSize = blockDim.x * gridDim.x * UNROLL;
  size_t offset = blockIdx.x * blockDim.x + threadIdx.x;

  // Fast path
  for (; offset + loopSize <= sizeInBP; offset += loopSize) {
    checkChunk<T>(ptr + offset);
  }
  // Slow path
  for (; offset < sizeInBP; offset += blockDim.x * gridDim.x) {
    BytePack tmp = ptr[offset];
    // if (isnan(tmp.f64[0]) || isnan(tmp.f64[1])) __trap();
    checkBytePack<T>(&tmp);
  }
}

// CHECK if a Tensor contains NAN in any of its element
void checkForNan(const at::Tensor& tensor, at::cuda::CUDAStream& stream) {
  // skip check for non float types
  if (!torch::is_floating_point(tensor)) {
    return;
  }
  const size_t maxNumThreadsPerBlock = 512;
  const size_t maxNumBlocks = 24;
  const size_t numThreadsPerBlock =
      std::min<size_t>(maxNumThreadsPerBlock, tensor.numel());

  const size_t numBlocks = std::min<size_t>(
      maxNumBlocks,
      (tensor.numel() + numThreadsPerBlock - 1) / numThreadsPerBlock);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      tensor.scalar_type(),
      "checkForNaN",
      [&] {
        checkForNaN<scalar_t><<<numBlocks, numThreadsPerBlock, 0, stream>>>(
            tensor.data_ptr<scalar_t>(), tensor.numel());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
}

} // namespace c10d

#endif // USE_C10D_NCCL
