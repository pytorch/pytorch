#include <gtest/gtest.h>

#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <c10/cuda/impl/CUDAGuardImpl.h>
#include <c10/cuda/impl/CUDATest.h>

using namespace c10::cuda::impl;

TEST(CUDATest, SmokeTest) {
  c10_cuda_test();
}

// Regression: the device-guard restore path is noexcept, but internally
// MaybeSetDevice() -> SetDevice() can throw -- via TORCH_CHECK(device >= 0) on
// the pre-CUDA-12 / HIP path (a pure host check), or via
// C10_CUDA_CHECK(cudaGetDevice) on a device carrying a sticky error. A leaked
// throw from a noexcept method calls std::terminate. A negative device index
// deterministically hits the TORCH_CHECK throw where MaybeSetDevice routes to
// SetDevice, so uncheckedSetDevice must swallow it (warn) instead of
// terminating -- if it regresses, this process aborts. On the CUDA-12 branch
// MaybeSetDevice short-circuits for an invalid index, so this is a benign
// no-op there.
TEST(CUDAGuardImplTest, UncheckedSetDeviceSwallowsErrorAndDoesNotTerminate) {
  CUDAGuardImpl impl;
  EXPECT_NO_THROW(
      impl.uncheckedSetDevice(c10::Device(c10::DeviceType::CUDA, -1)));
}
