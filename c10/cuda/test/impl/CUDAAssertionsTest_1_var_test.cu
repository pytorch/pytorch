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

void did_not_fail_diagnostics() {
  std::cerr
      << "c10::cuda::CUDAKernelLaunchRegistry::get_singleton_ref().enabled_at_runtime = "
      << c10::cuda::CUDAKernelLaunchRegistry::get_singleton_ref().enabled_at_runtime
      << std::endl;
  std::cerr
      << "c10::cuda::CUDAKernelLaunchRegistry::get_singleton_ref().enabled_at_compile_time = "
      << c10::cuda::CUDAKernelLaunchRegistry::get_singleton_ref().enabled_at_compile_time
      << std::endl;
  std::cerr
      << "c10::cuda::CUDAKernelLaunchRegistry::get_singleton_ref().do_all_devices_support_managed_memory = "
      << c10::cuda::CUDAKernelLaunchRegistry::get_singleton_ref()
             .do_all_devices_support_managed_memory
      << std::endl;
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
 * kernel used takes only 1 variable as parameter function.
 */
void cuda_device_assertions_1_var_test() {
  const auto stream = c10::cuda::getStreamFromPool();
  TORCH_DSA_KERNEL_LAUNCH(
      cuda_always_fail_assertion_kernel,
      1, /* Blocks */
      1, /* Threads */
      0, /* Shared mem */
      stream, /* Stream */
      1);

  try {
    c10::cuda::device_synchronize();
    did_not_fail_diagnostics();
    throw std::runtime_error("Test didn't fail, but should have.");
  } catch (const c10::Error& err) {
    const auto err_str = std::string(err.what());
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
            "Name of kernel launched that led to failure = cuda_always_fail_assertion_kernel"));
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

TEST(CUDATest, cuda_device_assertions_1_var_test) {
#ifdef TORCH_USE_CUDA_DSA
  c10::cuda::CUDAKernelLaunchRegistry::get_singleton_ref().enabled_at_runtime = true;
  did_not_fail_diagnostics();
  cuda_device_assertions_1_var_test();
#else
  GTEST_SKIP() << "CUDA device-side assertions (DSA) was not enabled at compile time.";
#endif
}
