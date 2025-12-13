#include <ATen/cuda/CUDAContextLight.h>
#include <ATen/cuda/Sleep.h>

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

namespace at::cuda {
namespace {
__global__ void spin_kernel(int64_t cycles) {
  // Few AMD specific GPUs have different clock intrinsic
#if defined(__GFX11__) && defined(USE_ROCM) && !defined(__CUDA_ARCH__)
  int64_t start_clock = wall_clock64();
#else
  // see concurrentKernels CUDA sampl
  int64_t start_clock = clock64();
#endif
  int64_t clock_offset = 0;
  while (clock_offset < cycles)
  {
#if defined(__GFX11__) && defined(USE_ROCM) && !defined(__CUDA_ARCH__)
    clock_offset = wall_clock64() - start_clock;
#else
    clock_offset = clock64() - start_clock;
#endif
  }
}

thread_local int *flag = nullptr;

__global__ void busy_wait_for_flag_kernel(int *flag) {
  atomicExch(flag, 1);
  while (atomicAdd(flag, 0) == 1) {
    // do nothing
  }
}

__global__ void clear_flag_kernel(int *flag) {
  atomicExch(flag, 0);
}

} // anonymous namespace

void sleep(int64_t cycles) {
  dim3 grid(1);
  dim3 block(1);
  spin_kernel<<<grid, block, 0, c10::cuda::getCurrentCUDAStream()>>>(cycles);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void busy_wait_for_flag() {
  if (!flag) {
    flag = (int*)c10::cuda::CUDACachingAllocator::raw_alloc(sizeof(int));
  }
  dim3 grid(1);
  dim3 block(1);
  busy_wait_for_flag_kernel<<<grid, block, 0, c10::cuda::getCurrentCUDAStream()>>>(flag);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void clear_flag() {
  if (!flag) {
    flag = (int*)c10::cuda::CUDACachingAllocator::raw_alloc(sizeof(int));
  }
  dim3 grid(1);
  dim3 block(1);
  clear_flag_kernel<<<grid, block, 0, c10::cuda::getCurrentCUDAStream()>>>(flag);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

#ifdef USE_ROCM
__global__ void flush_icache_kernel()
{
    asm __volatile__("s_icache_inv \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t" ::
                         :);
}
#endif

void flush_icache() {
#ifdef USE_ROCM
  dim3 grid(at::cuda::getCurrentDeviceProperties()->multiProcessorCount * 60);
  dim3 block(64);
  flush_icache_kernel<<<grid, block, 0, c10::cuda::getCurrentCUDAStream()>>>();
  C10_CUDA_KERNEL_LAUNCH_CHECK();
#endif
}

}  // namespace at::cuda
