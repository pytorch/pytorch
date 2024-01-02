#include <gtest/gtest.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDADevice.h>

TEST(CudaDeviceTest, getDeviceFromPtr_fails_with_host_memory) {
  if (!at::cuda::is_available()) {
    return;
  }

  int dummy = 0;

  ASSERT_THROW(at::cuda::getDeviceFromPtr(&dummy), c10::Error);
}
