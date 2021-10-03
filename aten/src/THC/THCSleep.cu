#include <THC/THCSleep.h>


__global__ void spin_kernel(int64_t cycles)
{
  // see concurrentKernels CUDA sampl
  int64_t start_clock = clock64();
  int64_t clock_offset = 0;
  while (clock_offset < cycles)
  {
    clock_offset = clock64() - start_clock;
  }
}

void THC_sleep(THCState* state, int64_t cycles)
{
  dim3 grid(1);
  dim3 block(1);
  spin_kernel<<<grid, block, 0, c10::cuda::getCurrentCUDAStream()>>>(cycles);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
