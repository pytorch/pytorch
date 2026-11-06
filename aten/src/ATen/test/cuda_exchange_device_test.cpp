#include <gtest/gtest.h>

#include <ATen/DeviceAccelerator.h>
#include <ATen/cuda/CUDAContext.h>


TEST(CudaExchangeDeviceTest, checkPrimaryContext) {
  if (!at::cuda::is_available()) {
    return;
  }

  // Before HIP 7.14, hipSetDevice does not create the primary context; it
  // appears on first real use of the device instead.
#if defined(USE_ROCM) && TORCH_HIP_VERSION < 714
  constexpr bool kSetDeviceCreatesContext = false;
#else
  constexpr bool kSetDeviceCreatesContext = true;
#endif

  ASSERT_FALSE(at::cuda::hasPrimaryContext(0));
  at::cuda::MaybeExchangeDevice(0);
  ASSERT_FALSE(at::cuda::hasPrimaryContext(0));
  at::accelerator::maybeExchangeDevice(0);
  ASSERT_FALSE(at::cuda::hasPrimaryContext(0));

  if (at::cuda::device_count() > 1) {
    ASSERT_FALSE(at::cuda::hasPrimaryContext(1));
    at::cuda::ExchangeDevice(1);
    ASSERT_EQ(at::cuda::hasPrimaryContext(1), kSetDeviceCreatesContext);
  }

  ASSERT_FALSE(at::cuda::hasPrimaryContext(0));
  at::cuda::MaybeExchangeDevice(0);
  ASSERT_FALSE(at::cuda::hasPrimaryContext(0));
  at::accelerator::maybeExchangeDevice(0);
  ASSERT_FALSE(at::cuda::hasPrimaryContext(0));
  at::accelerator::exchangeDevice(0);
  ASSERT_EQ(at::cuda::hasPrimaryContext(0), kSetDeviceCreatesContext);
}
