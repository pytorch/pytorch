#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

#include <ATen/ATen.h>

__global__ void tanh_add_kernel(
    const float* __restrict__ x,
    const float* __restrict__ y,
    float* __restrict__ output,
    const int size) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    const float tanh_x = 2.0f / (1.0f + __expf(-2.0f * x[index])) - 1;
    const float tanh_y = 2.0f / (1.0f + __expf(-2.0f * y[index])) - 1;
    output[index] = tanh_x + tanh_y;
  }
}

void tanh_add_cuda(const float* x, const float* y, float* output, int size) {
  const int threads = 1024;
  const int blocks = (size + threads - 1) / threads;
  tanh_add_kernel<<<blocks, threads>>>(x, y, output, size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
