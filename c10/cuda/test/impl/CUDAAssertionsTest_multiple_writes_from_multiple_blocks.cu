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

const auto max_assertions_failure_str =
    "Assertion failure " + std::to_string(C10_CUDA_DSA_ASSERTION_COUNT - 1);

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
 * TEST: Triggering device side assertion from multiple block but single thread
 * <<<10,1>>>. Here we are triggering assertion on 10 blocks, each with only 1
 * thread. Since we have more than 10 SM on a GPU, we expect each block to be
 * executed and successfully assert, Hence we will see assertions logged from
 * each block here.
 */
void cuda_device_assertions_multiple_writes_from_multiple_blocks() {
  const auto stream = c10::cuda::getStreamFromPool();
  TORCH_DSA_KERNEL_LAUNCH(
      cuda_always_fail_assertion_kernel,
      10, /* Blocks */
      1, /* Threads */
      0, /* Shared mem */
      stream, /* Stream */
      1);

  try {
    c10::cuda::device_synchronize();
    throw std::runtime_error("Test didn't fail, but should have.");
  } catch (const c10::Error& err) {
    const auto err_str = std::string(err.what());
    ASSERT_THAT(err_str, HasSubstr(max_assertions_failure_str));
    ASSERT_THAT(
        err_str, HasSubstr("Thread ID that failed assertion = [0,0,0]"));
    ASSERT_THAT(err_str, HasSubstr("Block ID that failed assertion = [0,0,0]"));
    ASSERT_THAT(err_str, HasSubstr("Block ID that failed assertion = [1,0,0]"));
    ASSERT_THAT(err_str, HasSubstr("Block ID that failed assertion = [2,0,0]"));
    ASSERT_THAT(err_str, HasSubstr("Block ID that failed assertion = [3,0,0]"));
    ASSERT_THAT(err_str, HasSubstr("Block ID that failed assertion = [4,0,0]"));
    ASSERT_THAT(err_str, HasSubstr("Block ID that failed assertion = [5,0,0]"));
    ASSERT_THAT(err_str, HasSubstr("Block ID that failed assertion = [6,0,0]"));
    ASSERT_THAT(err_str, HasSubstr("Block ID that failed assertion = [7,0,0]"));
    ASSERT_THAT(err_str, HasSubstr("Block ID that failed assertion = [8,0,0]"));
    ASSERT_THAT(err_str, HasSubstr("Block ID that failed assertion = [9,0,0]"));
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

TEST(CUDATest, cuda_device_assertions_multiple_writes_from_multiple_blocks) {
#ifdef TORCH_USE_CUDA_DSA
  c10::cuda::CUDAKernelLaunchRegistry::get_singleton_ref().enabled_at_runtime = true;
  cuda_device_assertions_multiple_writes_from_multiple_blocks();
#else
  GTEST_SKIP() << "CUDA device-side assertions (DSA) was not enabled at compile time.";
#endif
}
