#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

#include <ATen/ATen.h>

#include "cuda_dlink_extension_add.cuh"

__global__ void add_kernel(const float* a, const float* b, float* output, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    add(a + i, b + i, output + i);
  }
}

// output = a * b + c
void add_cuda(const float* a, const float* b, float* output, int size) {
  const int threads = 1024;
  const int blocks = (size + threads - 1) / threads;
  add_kernel<<<blocks, threads>>>(a, b, output, size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
