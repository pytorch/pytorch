#include "CUDATest.hpp"
#include <ATen/cuda/Exceptions.h>

namespace c10d {
namespace test {

namespace {
__global__ void waitClocks(const uint64_t count) {
  clock_t start = clock64();
  clock_t offset = 0;
  while (offset < count) {
    offset = clock() - start;
  }
}

} // namespace

void cudaSleep(at::cuda::CUDAStream& stream, uint64_t clocks) {
  waitClocks<<<1, 1, 0, stream.stream()>>>(clocks);
}

int cudaNumDevices() {
  int n = 0;
  AT_CUDA_CHECK(cudaGetDeviceCount(&n));
  return n;
}

} // namespace test
} // namespace c10d
