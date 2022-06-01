#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

#include <ATen/ATen.h>


__global__ void add_kernel(const float* a, const float* b, float* output, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    output[i] = a[i] + b[i];
  }
}

__global__ void mm_kernel(const float* a, const float* b, float* output, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    output[i] = a[i] * b[i];
  }
}

// output = a * b + c
__global__ void addmm_kernel(const float* a, const float* b, const float* c, float* output, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    mm_kernel<<<1, 1>>>(a + i, b + i, output + i, 1);
    add_kernel<<<1, 1>>>(output + i, c + i, output + i, 1);
  }
}

// output = a * b + c
void addmm_cuda(const float* a, const float* b, const float* c, float* output, int size) {
  const int threads = 1024;
  const int blocks = (size + threads - 1) / threads;
  addmm_kernel<<<blocks, threads>>>(a, b, c, output, size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
