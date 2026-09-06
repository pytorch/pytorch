#include <c10/core/AllocatorConfig.h>
#include <c10/cuda/CUDAAllocatorConfig.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Exception.h>

#include <gtest/gtest.h>

#include <atomic>
#include <string>

// Device tests for allocations that no expandable segment can ever satisfy,
// because a single allocation cannot span segments. They must fail through the
// allocator's normal out-of-memory path -- notifying OOM observers -- and not
// crash or bypass it. Live in their own binary because they mutate the
// process-global allocator config and tag streams.

namespace {

// Set by the OOM observer. A namespace-scope flag (rather than a captured
// local) because observers cannot be detached and outlive the test that
// attached them.
std::atomic<bool> g_oom_observed{false};

// Attaches the observer exactly once for the whole binary; every test then just
// resets the flag.
void attachOomObserverOnce() {
  static const bool attached = [] {
    c10::cuda::CUDACachingAllocator::attachOutOfMemoryObserver(
        [](int64_t, size_t, size_t, size_t) { g_oom_observed = true; });
    return true;
  }();
  (void)attached;
}

// Returns false (and skips) when the host has no GPU.
bool initDeviceOrSkip() {
  if (c10::cuda::device_count() == 0) {
    return false;
  }
  c10::cuda::set_device(0);
  c10::cuda::CUDACachingAllocator::init(c10::cuda::device_count());
  attachOomObserverOnce();
  g_oom_observed = false;
  return true;
}

// Runs an allocation expected to OOM and returns the error message.
std::string expectOomMessage(size_t request) {
  try {
    auto blk = c10::cuda::CUDACachingAllocator::get()->allocate(request);
  } catch (const c10::OutOfMemoryError& e) {
    return e.what();
  }
  ADD_FAILURE() << "expected an OutOfMemoryError for a " << request
                << "-byte allocation";
  return {};
}

size_t deviceTotalBytes() {
  cudaDeviceProp prop{};
  C10_CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  return prop.totalGlobalMem;
}

} // namespace

TEST(
    ExpandableSegmentReserveDeviceTest,
    DownsizedReserveAllocLargerThanReserve) {
  if (!initDeviceOrSkip()) {
    GTEST_SKIP() << "no CUDA device";
  }
  const size_t device_total = deviceTotalBytes();

  // Reserve ~5% of the device for the tagged class, then ask for 4x that. The
  // request is far below total device memory, so it fails only because a single
  // allocation cannot span expandable segments.
  c10::CachingAllocator::setAllocatorSettings(
      "expandable_segments:True,"
      "expandable_segments_reserve_by_class:[tiny:0.05]");
  ASSERT_TRUE(c10::cuda::CUDACachingAllocator::CUDAAllocatorConfig::
                  expandable_segments())
      << "expandable_segments must be enabled for this test to be meaningful";

  auto stream = c10::cuda::getCurrentCUDAStream(0);
  c10::cuda::CUDACachingAllocator::setExpandableSegmentReserveClassForStream(
      stream.stream(), "tiny");

  const size_t request = (device_total / 20) * 4;
  ASSERT_LT(request, device_total)
      << "request must be satisfiable if segments could span";

  const std::string msg = expectOomMessage(request);
  // The allocation must fail through the normal OOM path, not a bespoke throw
  // that skips observers, the OOM counter and the release-and-retry step.
  EXPECT_TRUE(g_oom_observed)
      << "out-of-memory observers must be notified for an unsatisfiable "
         "allocation";
  // ...and that path must explain why an allocation failed on a device that
  // still has plenty of memory free.
  EXPECT_NE(msg.find("expandable-segment"), std::string::npos) << msg;
  EXPECT_NE(msg.find("expandable_segments_reserve_by_class"), std::string::npos)
      << msg;
  EXPECT_NE(msg.find("tiny"), std::string::npos) << msg;

  c10::cuda::CUDACachingAllocator::setExpandableSegmentReserveClassForStream(
      stream.stream(), "");
}

TEST(ExpandableSegmentReserveDeviceTest, GlobalReserveUntaggedStream) {
  if (!initDeviceOrSkip()) {
    GTEST_SKIP() << "no CUDA device";
  }
  const size_t device_total = deviceTotalBytes();

  // Same failure via the global knob on an untagged stream -- i.e. what a user
  // hits with only PYTORCH_CUDA_ALLOC_CONF set and no code change.
  c10::CachingAllocator::setAllocatorSettings(
      "expandable_segments:True,expandable_segments_reserve:0.05");

  // A stream with no reserve class resolves to the global reserve. Clear the
  // tag explicitly so this does not depend on another test having cleaned up.
  auto stream = c10::cuda::getCurrentCUDAStream(0);
  c10::cuda::CUDACachingAllocator::setExpandableSegmentReserveClassForStream(
      stream.stream(), "");
  ASSERT_EQ(
      c10::cuda::CUDACachingAllocator::
          getExpandableSegmentReserveClassForStream(stream.stream()),
      "");

  const size_t request = (device_total / 20) * 4;
  const std::string msg = expectOomMessage(request);
  EXPECT_TRUE(g_oom_observed)
      << "out-of-memory observers must be notified for an unsatisfiable "
         "allocation";
  EXPECT_NE(msg.find("expandable-segment"), std::string::npos) << msg;
  EXPECT_NE(msg.find("expandable_segments_reserve"), std::string::npos) << msg;
}

TEST(ExpandableSegmentReserveDeviceTest, OversizedAllocNotifiesOomObserver) {
  if (!initDeviceOrSkip()) {
    GTEST_SKIP() << "no CUDA device";
  }
  const size_t device_total = deviceTotalBytes();

  // Regression guard for the no-downsizing path: an allocation larger than the
  // whole device exceeds even the default 9/8 reserve, so it takes the same
  // "no segment can ever fit this" branch. It must still behave exactly like a
  // stock OOM (this is the C++ analogue of test_cuda.py's test_notifies_oom,
  // which caught an earlier revision that threw from inside the allocator and
  // silently skipped the observers).
  c10::CachingAllocator::setAllocatorSettings("expandable_segments:True");

  const size_t request = device_total * 100;
  const std::string msg = expectOomMessage(request);
  EXPECT_TRUE(g_oom_observed)
      << "out-of-memory observers must be notified for an allocation larger "
         "than the device";
  // The reserve is not why this failed -- the request does not fit on the
  // device at all -- so the message must not send the user chasing the reserve
  // knobs.
  EXPECT_EQ(msg.find("expandable-segment"), std::string::npos) << msg;
}
