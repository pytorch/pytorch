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
 * <<<10,128>>>. Here we are triggering assertion on 10 blocks, each with only
 * 128 thread.
 */
void cuda_device_assertions_multiple_writes_from_blocks_and_threads() {
  bool run_threads = false;

  // Create a function to launch kernel that waits for a signal, to try to
  // ensure everything is happening simultaneously
  const auto launch_the_kernel = [&]() {
    // Busy loop waiting for the signal to go
    while (!run_threads) {
    }

    TORCH_DSA_KERNEL_LAUNCH(
        cuda_always_fail_assertion_kernel,
        10, /* Blocks */
        128, /* Threads */
        0, /* Shared mem */
        c10::cuda::getCurrentCUDAStream(), /* Stream */
        1);
  };

  // Spin up a bunch of busy-looping threads
  std::vector<std::thread> threads;
  for (int i = 0; i < 10; i++) {
    threads.emplace_back(launch_the_kernel);
  }

  // Paranoid - wait for all the threads to get setup
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  // Mash
  run_threads = true;

  // Clean-up
  for (auto& x : threads) {
    x.join();
  }

  try {
    c10::cuda::device_synchronize();
    throw std::runtime_error("Test didn't fail, but should have.");
  } catch (const c10::Error& err) {
    const auto err_str = std::string(err.what());
    ASSERT_THAT(err_str, HasSubstr(max_assertions_failure_str));
    ASSERT_THAT(err_str, HasSubstr("Device that launched kernel = 0"));
    ASSERT_THAT(
        err_str,
        HasSubstr(
            "Name of kernel launched that led to failure = cuda_always_fail_assertion_kernel"));
    ASSERT_THAT(
        err_str, HasSubstr("File containing kernel launch = " __FILE__));
  }
}

TEST(CUDATest, cuda_device_assertions_multiple_writes_from_blocks_and_threads) {
#ifdef TORCH_USE_CUDA_DSA
  c10::cuda::CUDAKernelLaunchRegistry::get_singleton_ref().enabled_at_runtime = true;
  cuda_device_assertions_multiple_writes_from_blocks_and_threads();
#else
  GTEST_SKIP() << "CUDA device-side assertions (DSA) was not enabled at compile time.";
#endif
}
