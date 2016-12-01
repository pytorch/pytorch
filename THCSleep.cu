#include "THCSleep.h"


__global__ void spin_kernel(long long cycles)
{
  // see concurrentKernels CUDA sampl
  long long start_clock = clock64();
  long long clock_offset = 0;
  while (clock_offset < cycles)
  {
    clock_offset = clock64() - start_clock;
  }
}

THC_API void THC_sleep(THCState* state, long long cycles)
{
  dim3 grid(1);
  dim3 block(1);
  spin_kernel<<<grid, block, 0, THCState_getCurrentStream(state)>>>(cycles);
  THCudaCheck(cudaGetLastError());
}
