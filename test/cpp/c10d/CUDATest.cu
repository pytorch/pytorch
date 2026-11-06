#include "CUDATest.hpp"
#include <ATen/cuda/Exceptions.h>

namespace c10d {
namespace test {

namespace {
__global__ void waitClocks(const uint64_t count) {
  // On gfx11/gfx12, clock64() reads a per-CU counter that follows the shader
  // clock (not synced across CUs on gfx1250); wall_clock64() has a fixed rate.
#if (defined(__GFX11__) || defined(__GFX12__)) && defined(USE_ROCM) && !defined(__CUDA_ARCH__)
  clock_t start = wall_clock64();
#else
  clock_t start = clock64();
#endif
  clock_t offset = 0;
  while (offset < count) {
#if (defined(__GFX11__) || defined(__GFX12__)) && defined(USE_ROCM) && !defined(__CUDA_ARCH__)
    offset = wall_clock64() - start;
#else
    offset = clock() - start;
#endif
  }
}

} // namespace

void cudaSleep(at::cuda::CUDAStream& stream, uint64_t clocks) {
  waitClocks<<<1, 1, 0, stream.stream()>>>(clocks);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

int cudaNumDevices() {
  int n = 0;
  C10_CUDA_CHECK_WARN(cudaGetDeviceCount(&n));
  return n;
}

} // namespace test
} // namespace c10d
