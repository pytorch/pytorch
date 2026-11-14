#include <c10/core/AllocatorConfig.h>
#include <c10/cuda/CUDAAllocatorConfig.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Exception.h>
#include <c10/util/ScopeExit.h>

#include <gtest/gtest.h>

#if (defined(USE_ROCM) || defined(__HIP_PLATFORM_AMD__)) && defined(__linux__)
#define C10_TEST_HIP_VMM_INTERPOSITION
#include <dlfcn.h>
#include <hip/hip_runtime_api.h>
#endif

#include <array>
#include <atomic>
#include <string>

// Device tests for allocations that no expandable segment can ever satisfy,
// because a single allocation cannot span segments. They must fail through the
// allocator's normal out-of-memory path -- notifying OOM observers -- and not
// crash or bypass it. Live in their own binary because they mutate the
// process-global allocator config and tag streams.

#ifdef C10_TEST_HIP_VMM_INTERPOSITION
namespace {
enum class InjectedVmmFailure : uint8_t { None, Map, SetAccess };

std::atomic<InjectedVmmFailure> g_injected_vmm_failure{
    InjectedVmmFailure::None};

bool takeInjectedVmmFailure(InjectedVmmFailure failure) {
  auto expected = failure;
  return g_injected_vmm_failure.compare_exchange_strong(
      expected, InjectedVmmFailure::None);
}
} // namespace

extern "C" hipError_t hipMemMap(
    void* ptr,
    size_t size,
    size_t offset,
    hipMemGenericAllocationHandle_t handle,
    unsigned long long flags) {
  if (takeInjectedVmmFailure(InjectedVmmFailure::Map)) {
    return hipErrorInvalidValue;
  }
  using HipMemMap = hipError_t (*)(
      void*,
      size_t,
      size_t,
      hipMemGenericAllocationHandle_t,
      unsigned long long);
  static const auto real_hip_mem_map =
      reinterpret_cast<HipMemMap>(dlsym(RTLD_NEXT, "hipMemMap"));
  return real_hip_mem_map ? real_hip_mem_map(ptr, size, offset, handle, flags)
                          : hipErrorUnknown;
}

extern "C" hipError_t hipMemSetAccess(
    void* ptr,
    size_t size,
    const hipMemAccessDesc* desc,
    size_t count) {
  if (takeInjectedVmmFailure(InjectedVmmFailure::SetAccess)) {
    return hipErrorInvalidValue;
  }
  using HipMemSetAccess =
      hipError_t (*)(void*, size_t, const hipMemAccessDesc*, size_t);
  static const auto real_hip_mem_set_access =
      reinterpret_cast<HipMemSetAccess>(dlsym(RTLD_NEXT, "hipMemSetAccess"));
  return real_hip_mem_set_access
      ? real_hip_mem_set_access(ptr, size, desc, count)
      : hipErrorUnknown;
}
#endif

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

#ifdef C10_TEST_HIP_VMM_INTERPOSITION
void expectVmmFailureIsRetryable(InjectedVmmFailure failure) {
  c10::CachingAllocator::setAllocatorSettings("expandable_segments:True");
  const size_t segment_size =
      c10::CachingAllocator::AcceleratorAllocatorConfig::large_segment_size();
  c10::cuda::CUDACachingAllocator::emptyCache();
  const auto stream = c10::cuda::getStreamFromPool(false, 0);

  std::array<void*, 3> blocks{};
  auto release_blocks = c10::make_scope_exit([&] {
    for (auto& block : blocks) {
      if (block) {
        c10::cuda::CUDACachingAllocator::raw_delete(block);
        block = nullptr;
      }
    }
    c10::cuda::CUDACachingAllocator::emptyCache();
  });
  for (auto& block : blocks) {
    block = c10::cuda::CUDACachingAllocator::raw_alloc_with_stream(
        segment_size, stream.stream());
  }
  ASSERT_EQ(
      static_cast<char*>(blocks[1]) - static_cast<char*>(blocks[0]),
      segment_size);
  ASSERT_EQ(
      static_cast<char*>(blocks[2]) - static_cast<char*>(blocks[1]),
      segment_size);

  void* hole = blocks[1];
  c10::cuda::CUDACachingAllocator::raw_delete(hole);
  blocks[1] = nullptr;
  c10::cuda::CUDACachingAllocator::emptyCache();

  g_injected_vmm_failure = failure;
  auto reset_injected_failure = c10::make_scope_exit(
      [] { g_injected_vmm_failure = InjectedVmmFailure::None; });
  EXPECT_THROW(
      blocks[1] = c10::cuda::CUDACachingAllocator::raw_alloc_with_stream(
          segment_size, stream.stream()),
      c10::Error);
  EXPECT_EQ(g_injected_vmm_failure, InjectedVmmFailure::None);
  ASSERT_EQ(blocks[1], nullptr);

  blocks[1] = c10::cuda::CUDACachingAllocator::raw_alloc_with_stream(
      segment_size, stream.stream());
  EXPECT_EQ(blocks[1], hole);
}
#endif

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

TEST(ExpandableSegmentReserveDeviceTest, RocmMapFailureIsRetryable) {
#ifdef C10_TEST_HIP_VMM_INTERPOSITION
  if (!initDeviceOrSkip()) {
    GTEST_SKIP() << "no ROCm device";
  }
  expectVmmFailureIsRetryable(InjectedVmmFailure::Map);
#else
  GTEST_SKIP() << "requires Linux HIP VMM interposition";
#endif
}

TEST(ExpandableSegmentReserveDeviceTest, RocmSetAccessFailureIsRetryable) {
#ifdef C10_TEST_HIP_VMM_INTERPOSITION
  if (!initDeviceOrSkip()) {
    GTEST_SKIP() << "no ROCm device";
  }
  expectVmmFailureIsRetryable(InjectedVmmFailure::SetAccess);
#else
  GTEST_SKIP() << "requires Linux HIP VMM interposition";
#endif
}
