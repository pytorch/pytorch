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
 * Device kernel that takes a single integer parameter as argument and
 * will never trigger a device side assertion.
 */
__global__ void cuda_always_succeed_assertion_kernel(
    const int a,
    TORCH_DSA_KERNEL_ARGS) {
  CUDA_KERNEL_ASSERT2(a == a);
}

// Windows doesn't like `fork`
#ifndef _MSC_VER
/**
 * TEST: Triggering device side assertion from 2 different processes from CPU.
 * The following code is testing if two processes from CPU that are running
 * GPU kernels (not necessarily simultaneously) and are asserting & writing
 * to the respective UVMs, mess up anything for each other.
 * Once parent process's kernel launch fails and causes a device-side assertion
 * and is still alive when the second process is interacting with the GPU,
 * trying to launch another kernel.
 */
void cuda_device_assertions_from_2_processes() {
  const auto n1 = fork();
  if (n1 == 0) {
    // This is the parent process, that will call an assertion failure.
    // This should execute before the child process.
    // We are achieving this by putting the child process to sleep.
    TORCH_DSA_KERNEL_LAUNCH(
        cuda_always_fail_assertion_kernel,
        1, /* Blocks */
        1, /* Threads */
        0, /* Shared mem */
        c10::cuda::getStreamFromPool(), /* Stream */
        1);
    try {
      c10::cuda::device_synchronize();
      throw std::runtime_error("Test didn't fail, but should have.");
    } catch (const c10::Error& err) {
      const auto err_str = std::string(err.what());
      ASSERT_THAT(
          err_str,
          HasSubstr(
              "1 CUDA device-side assertion failures were found on GPU #0!"));
    }
    // Keep this alive so we can see what happened to the other process
    std::this_thread::sleep_for(std::chrono::milliseconds(3000));
  } else {
    // This is the child process
    // We put it to sleep for next 2 seconds, to make sure that the parent has
    // asserted a failure already.
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    TORCH_DSA_KERNEL_LAUNCH(
        cuda_always_succeed_assertion_kernel,
        1, /* Blocks */
        1, /* Threads */
        0, /* Shared mem */
        c10::cuda::getStreamFromPool(), /* Stream */
        1);
    try {
      c10::cuda::device_synchronize();
    } catch (const c10::Error& err) {
      ASSERT_TRUE(false); // This kernel should not have failed, but did.
    }
    // End the child process
    exit(0);
  }
}

TEST(CUDATest, cuda_device_assertions_from_2_processes) {
#ifdef TORCH_USE_CUDA_DSA
  c10::cuda::CUDAKernelLaunchRegistry::get_singleton_ref().enabled_at_runtime = true;
  cuda_device_assertions_from_2_processes();
#else
  GTEST_SKIP() << "CUDA device-side assertions (DSA) was not enabled at compile time.";
#endif
}

#else

#endif
