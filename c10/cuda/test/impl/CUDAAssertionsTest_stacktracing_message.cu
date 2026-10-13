#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <c10/cuda/CUDADeviceAssertion.h>
#include <c10/cuda/CUDADeviceAssertionHost.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>

#include <stdexcept>
#include <string>

using ::testing::HasSubstr;
using ::testing::Not;

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
 * TEST: The device-side assertion diagnostic's "Backtrace of kernel launch
 * site" line must reflect the actual state of the stacktracing flag
 * (CUDAKernelLaunchRegistry::gather_launch_stacktrace, from
 * PYTORCH_CUDA_DSA_STACKTRACING).
 *
 * Regression test for an inverted branch that reported "Launch stacktracing
 * disabled." while stacktracing was enabled, and printed an empty (never
 * collected) backtrace while it was disabled.
 *
 * One device-side assertion failure per process, because a failed assertion
 * leaves the CUDA context unusable. The flag is read when the diagnostic is
 * formatted, so a single failure is enough to exercise both directions by
 * re-formatting the message with the flag toggled.
 */
void cuda_device_assertions_stacktracing_message() {
  if (c10::cuda::device_count() == 0) {
    GTEST_SKIP() << "no CUDA device";
  }
  auto& registry = c10::cuda::CUDAKernelLaunchRegistry::get_singleton_ref();
  if (!registry.do_all_devices_support_managed_memory) {
    GTEST_SKIP() << "Device-side assertions disabled because not all devices "
                    "support managed memory.";
  }
  registry.enabled_at_runtime = true;
  // Capture the backtrace at launch time so the enabled branch has something
  // to print.
  registry.gather_launch_stacktrace = true;

  try {
    TORCH_DSA_KERNEL_LAUNCH(
        cuda_always_fail_assertion_kernel,
        1, /* Blocks */
        1, /* Threads */
        0, /* Shared mem */
        c10::cuda::getStreamFromPool(), /* Stream */
        1);
    c10::cuda::device_synchronize();
    // The catch below is for c10::Error; a raw throw here must not be
    // catchable as c10::Error so a silently-passing kernel fails the test.
    // @allow-raw-throw: must not be catchable as c10::Error
    throw std::runtime_error("Test didn't fail, but should have.");
  } catch (const c10::Error&) {
    // Expected: the failing kernel surfaces as a device-side assertion.
  }

  // With stacktracing enabled the diagnostic must NOT claim it is disabled,
  // and must emit the launch-site backtrace section.
  const auto enabled_msg = c10::cuda::c10_retrieve_device_side_assertion_info();
  EXPECT_THAT(enabled_msg, HasSubstr("Backtrace of kernel launch site = "));
  EXPECT_THAT(enabled_msg, Not(HasSubstr("Launch stacktracing disabled.")));

  // With stacktracing disabled it must say so.
  registry.gather_launch_stacktrace = false;
  const auto disabled_msg =
      c10::cuda::c10_retrieve_device_side_assertion_info();
  EXPECT_THAT(disabled_msg, HasSubstr("Backtrace of kernel launch site = "));
  EXPECT_THAT(disabled_msg, HasSubstr("Launch stacktracing disabled."));
}

TEST(CUDATest, cuda_device_assertions_stacktracing_message) {
#ifdef TORCH_USE_CUDA_DSA
  c10::cuda::CUDAKernelLaunchRegistry::get_singleton_ref().enabled_at_runtime =
      true;
  cuda_device_assertions_stacktracing_message();
#else
  GTEST_SKIP() << "CUDA device-side assertions (DSA) was not enabled at compile "
                  "time.";
#endif
}
