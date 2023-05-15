#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <c10/cuda/CUDADeviceAssertion.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>

#include <chrono>
#include <iostream>
#include <string>
#include <thread>

using ::testing::HasSubstr;

/**
 * Device kernel that takes multiple integer parameters as arguments and
 * will always trigger a device side assertion.
 */
__global__ void cuda_multiple_vars_always_fail_assertion_kernel(
    const int a,
    const int b,
    const int c,
    const int d,
    TORCH_DSA_KERNEL_ARGS) {
  int i = a + b + c + d;
  if (i != 0) {
    CUDA_KERNEL_ASSERT2(i == -i);
  } else {
    CUDA_KERNEL_ASSERT2(i == i + 1);
  }
}

/**
 * Device kernel that takes a single integer parameter as argument and
 * will always trigger a device side assertion.
 */
__global__ void cuda_always_fail_assertion_kernel(
    const int a,
    TORCH_DSA_KERNEL_ARGS) {
  CUDA_KERNEL_ASSERT2(a != a);
}

/**
 * TEST: Triggering device side assertion on a simple <<<1,1>>> config.
 * kernel used takes multiple variables as parameters to the function.
 */
void cuda_device_assertions_catches_stream() {
  const auto stream = c10::cuda::getStreamFromPool();
  TORCH_DSA_KERNEL_LAUNCH(
      cuda_multiple_vars_always_fail_assertion_kernel,
      1, /* Blocks */
      1, /* Threads */
      0, /* Shared mem */
      stream, /* Stream */
      1, /* const int a */
      2, /* const int b */
      3, /* const int c */
      4 /* const int d */
  );

  try {
    c10::cuda::device_synchronize();
    throw std::runtime_error("Test didn't fail, but should have.");
  } catch (const c10::Error& err) {
    const auto err_str = std::string(err.what());
    ASSERT_THAT(
        err_str, HasSubstr("# of GPUs this process interacted with = 1"));
    ASSERT_THAT(
        err_str,
        HasSubstr("CUDA device-side assertion failures were found on GPU #0!"));
    ASSERT_THAT(
        err_str, HasSubstr("Thread ID that failed assertion = [0,0,0]"));
    ASSERT_THAT(err_str, HasSubstr("Block ID that failed assertion = [0,0,0]"));
    ASSERT_THAT(err_str, HasSubstr("Device that launched kernel = 0"));
    ASSERT_THAT(
        err_str,
        HasSubstr(
            "Name of kernel launched that led to failure = cuda_multiple_vars_always_fail_assertion_kernel"));
    ASSERT_THAT(
        err_str, HasSubstr("File containing kernel launch = " __FILE__));
    ASSERT_THAT(
        err_str,
        HasSubstr(
            "Function containing kernel launch = " +
            std::string(__FUNCTION__)));
    ASSERT_THAT(
        err_str,
        HasSubstr(
            "Stream kernel was launched on = " + std::to_string(stream.id())));
  }
}

TEST(CUDATest, cuda_device_assertions_catches_stream) {
#ifdef TORCH_USE_CUDA_DSA
  c10::cuda::CUDAKernelLaunchRegistry::get_singleton_ref().enabled_at_runtime = true;
  cuda_device_assertions_catches_stream();
#else
  GTEST_SKIP() << "CUDA device-side assertions (DSA) was not enabled at compile time.";
#endif
}
