#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGraphMemory.h>
#include <c10/cuda/impl/CUDAGraphMemory.h>
#include <c10/util/ScopeExit.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace {

namespace allocator = c10::cuda::CUDACachingAllocator;
namespace graph = c10::cuda::CUDAGraphMemory;

// No method performs device work. Strict expectations reject unexpected calls.
class MockLegacyAllocator : public allocator::CUDAAllocator {
 public:
  using CUDAAllocator::recordStream;

  MOCK_METHOD(c10::DataPtr, allocate, (size_t), (override));
  MOCK_METHOD(void, copy_data, (void*, const void*, size_t), (const, override));
  MOCK_METHOD(void*, raw_alloc, (size_t), (override));
  MOCK_METHOD(void*, raw_alloc_with_stream, (size_t, cudaStream_t), (override));
  MOCK_METHOD(void, raw_delete, (void*), (override));
  MOCK_METHOD(void, init, (int), (override));
  MOCK_METHOD(bool, initialized, (), (override));
  MOCK_METHOD(void, emptyCache, (c10::MempoolId_t), (override));
  MOCK_METHOD(double, getMemoryFraction, (c10::DeviceIndex), (override));
  MOCK_METHOD(void, setMemoryFraction, (double, c10::DeviceIndex), (override));
  MOCK_METHOD(
      std::vector<allocator::StreamSegmentSize>,
      getExpandableSegmentSizes,
      (c10::DeviceIndex),
      (override));
  MOCK_METHOD(void, enable, (bool), (override));
  MOCK_METHOD(bool, isEnabled, (), (const, override));
  MOCK_METHOD(void, cacheInfo, (c10::DeviceIndex, size_t*), (override));
  MOCK_METHOD(void*, getBaseAllocation, (void*, size_t*), (override));
  MOCK_METHOD(
      void,
      recordStream,
      (const c10::DataPtr&, c10::cuda::CUDAStream),
      (override));
  MOCK_METHOD(
      c10::CachingDeviceAllocator::DeviceStats,
      getDeviceStats,
      (c10::DeviceIndex),
      (override));
  MOCK_METHOD(void, resetAccumulatedStats, (c10::DeviceIndex), (override));
  MOCK_METHOD(void, resetPeakStats, (c10::DeviceIndex), (override));
  MOCK_METHOD(
      allocator::SnapshotInfo,
      snapshot,
      (c10::MempoolId_t, bool),
      (override));
  MOCK_METHOD(
      void,
      beginAllocateToPool,
      (c10::DeviceIndex, c10::MempoolId_t, std::function<bool(cudaStream_t)>),
      (override));
  MOCK_METHOD(
      void,
      endAllocateToPool,
      (c10::DeviceIndex, c10::MempoolId_t),
      (override));
  MOCK_METHOD(void, markCaptureBegin, (c10::DeviceIndex), (override));
  MOCK_METHOD(void, markCaptureEnd, (c10::DeviceIndex), (override));
  MOCK_METHOD(
      void,
      releasePool,
      (c10::DeviceIndex, c10::MempoolId_t),
      (override));
  MOCK_METHOD(allocator::ShareableHandle, shareIpcHandle, (void*), (override));
  MOCK_METHOD(std::shared_ptr<void>, getIpcDevPtr, (std::string), (override));
  MOCK_METHOD(
      void,
      recordHistory,
      (bool,
       allocator::CreateContextFn,
       size_t,
       allocator::RecordContext,
       bool,
       const std::vector<std::string>&),
      (override));
  MOCK_METHOD(
      void,
      attachOutOfMemoryObserver,
      (allocator::OutOfMemoryObserver),
      (override));
  MOCK_METHOD(
      void,
      attachOomRejectionObserver,
      (allocator::OomRejectionObserver),
      (override));
  MOCK_METHOD(
      void,
      attachAllocatorTraceTracker,
      (allocator::AllocatorTraceTracker),
      (override));
  MOCK_METHOD(
      void,
      enablePeerAccess,
      (c10::DeviceIndex, c10::DeviceIndex),
      (override));
  MOCK_METHOD(
      cudaError_t,
      memcpyAsync,
      (void*, int, const void*, int, size_t, cudaStream_t, bool),
      (override));
  MOCK_METHOD(
      std::shared_ptr<allocator::AllocatorState>,
      getCheckpointState,
      (c10::DeviceIndex, c10::MempoolId_t),
      (override));
  MOCK_METHOD(
      allocator::CheckpointDelta,
      setCheckpointPoolState,
      (c10::DeviceIndex, std::shared_ptr<allocator::AllocatorState>),
      (override));
  MOCK_METHOD(std::string, name, (), (override));
};

class MockRegisteredAllocator : public MockLegacyAllocator {
 public:
  MOCK_METHOD(
      void,
      markCaptureBeginWithRegistration,
      (c10::DeviceIndex, const graph::CaptureRegistration&),
      (override));
  MOCK_METHOD(
      size_t,
      markCaptureEndWithCaptureId,
      (c10::DeviceIndex, c10::cuda::CaptureId_t),
      (override));
};

auto throwsError(const char* message) {
  return testing::ThrowsMessage<c10::Error>(testing::HasSubstr(message));
}

} // namespace

TEST(CUDAGraphMemoryDispatchTest, LegacyCallbacksReceiveTheDeviceExactlyOnce) {
  testing::StrictMock<MockLegacyAllocator> backend;
  constexpr c10::DeviceIndex kDevice = 3;
  constexpr c10::cuda::CaptureId_t kCapture = 42;
  const graph::CaptureRegistration registration{kCapture, std::nullopt};
  {
    testing::InSequence order;
    EXPECT_CALL(backend, markCaptureBegin(kDevice)).Times(1);
    EXPECT_CALL(backend, markCaptureEnd(kDevice)).Times(1);
  }
  auto* previous = allocator::allocator.exchange(&backend);
  auto restore =
      c10::make_scope_exit([&] { allocator::allocator.store(previous); });

  graph::markCaptureBegin(kDevice, registration);
  EXPECT_EQ(graph::markCaptureEnd(kDevice, kCapture), 0u);
}

TEST(
    CUDAGraphMemoryDispatchTest,
    RegisteredCallbacksReceiveAllMetadataAndResult) {
  testing::StrictMock<MockRegisteredAllocator> backend;
  constexpr c10::DeviceIndex kDevice = 5;
  constexpr c10::cuda::CaptureId_t kRoot = 41;
  constexpr c10::cuda::CaptureId_t kChild = 42;
  constexpr size_t kInvalidFrees = 7;
  std::array<char, 2> storage{};
  const auto parentStream = reinterpret_cast<cudaStream_t>(&storage[0]);
  const auto childStream = reinterpret_cast<cudaStream_t>(&storage[1]);
  const graph::CaptureRegistration registration{
      kChild, kRoot, childStream, parentStream};
  graph::CaptureRegistration received;
  {
    testing::InSequence order;
    EXPECT_CALL(backend, markCaptureBeginWithRegistration(kDevice, testing::_))
        .Times(1)
        .WillOnce(testing::SaveArg<1>(&received));
    EXPECT_CALL(backend, markCaptureEndWithCaptureId(kDevice, kChild))
        .Times(1)
        .WillOnce(testing::Return(kInvalidFrees));
  }
  auto* previous = allocator::allocator.exchange(&backend);
  auto restore =
      c10::make_scope_exit([&] { allocator::allocator.store(previous); });

  graph::markCaptureBegin(kDevice, registration);
  EXPECT_EQ(received.capture_id, kChild);
  EXPECT_EQ(received.parent_capture_id, std::optional{kRoot});
  EXPECT_EQ(received.primary_stream, childStream);
  EXPECT_EQ(received.parent_dependency_stream, parentStream);
  EXPECT_EQ(graph::markCaptureEnd(kDevice, kChild), kInvalidFrees);
}

TEST(CUDAGraphMemoryTrackerTest, LegacyCaptureCountBalances) {
  graph::CaptureTracker tracker;
  EXPECT_FALSE(tracker.hasActiveCaptures());
  EXPECT_EQ(tracker.activeCaptureCount(), 0);
  tracker.captureBegin();
  tracker.captureBegin();
  EXPECT_TRUE(tracker.hasActiveCaptures());
  EXPECT_EQ(tracker.activeCaptureCount(), 2);
  tracker.captureEnd();
  EXPECT_EQ(tracker.activeCaptureCount(), 1);
  tracker.captureEnd();
  EXPECT_FALSE(tracker.hasActiveCaptures());
  EXPECT_EQ(tracker.activeCaptureCount(), 0);
  EXPECT_THAT(
      [&] { tracker.captureEnd(); },
      throwsError("captureEnd called with no active capture"));
  EXPECT_EQ(tracker.activeCaptureCount(), 0);
}

TEST(CUDAGraphMemoryTrackerTest, ConditionalAliasesFollowTheParentAndExpire) {
  graph::CaptureTracker tracker;
  constexpr c10::cuda::CaptureId_t kRoot = 1;
  constexpr c10::cuda::CaptureId_t kChild = 2;
  constexpr c10::cuda::CaptureId_t kGrandchild = 3;
  std::array<char, 3> storage{};
  const auto rootStream = reinterpret_cast<cudaStream_t>(&storage[0]);
  const auto childStream = reinterpret_cast<cudaStream_t>(&storage[1]);
  const auto grandchildStream = reinterpret_cast<cudaStream_t>(&storage[2]);
  tracker.captureBegin({kRoot, std::nullopt, rootStream});
  tracker.captureBegin({kChild, kRoot, childStream, rootStream});
  tracker.captureBegin({kGrandchild, kChild, grandchildStream, childStream});
  EXPECT_EQ(tracker.activeCaptureCount(), 3);
  EXPECT_EQ(tracker.reuseStream(childStream), rootStream);
  EXPECT_EQ(tracker.reuseStream(grandchildStream), rootStream);

  EXPECT_EQ(tracker.captureEnd(kGrandchild), 0u);
  EXPECT_EQ(tracker.activeCaptureCount(), 2);
  EXPECT_EQ(tracker.reuseStream(grandchildStream), grandchildStream);
  EXPECT_EQ(tracker.reuseStream(childStream), rootStream);
  EXPECT_EQ(tracker.captureEnd(kChild), 0u);
  EXPECT_EQ(tracker.activeCaptureCount(), 1);
  EXPECT_EQ(tracker.reuseStream(childStream), childStream);
  EXPECT_EQ(tracker.captureEnd(kRoot), 0u);
  EXPECT_FALSE(tracker.hasActiveCaptures());
}

TEST(CUDAGraphMemoryTrackerTest, RootTeardownPreservesAnUnrelatedCaptureTree) {
  graph::CaptureTracker tracker;
  constexpr c10::cuda::CaptureId_t kRoot = 1;
  constexpr c10::cuda::CaptureId_t kChild = 2;
  constexpr c10::cuda::CaptureId_t kGrandchild = 3;
  constexpr c10::cuda::CaptureId_t kOtherRoot = 4;
  constexpr c10::cuda::CaptureId_t kOtherChild = 5;
  std::array<char, 5> storage{};
  const auto rootStream = reinterpret_cast<cudaStream_t>(&storage[0]);
  const auto childStream = reinterpret_cast<cudaStream_t>(&storage[1]);
  const auto grandchildStream = reinterpret_cast<cudaStream_t>(&storage[2]);
  const auto otherRootStream = reinterpret_cast<cudaStream_t>(&storage[3]);
  const auto otherChildStream = reinterpret_cast<cudaStream_t>(&storage[4]);
  tracker.captureBegin({kRoot, std::nullopt, rootStream});
  tracker.captureBegin({kChild, kRoot, childStream, rootStream});
  tracker.captureBegin({kGrandchild, kChild, grandchildStream, childStream});
  tracker.captureBegin({kOtherRoot, std::nullopt, otherRootStream});
  tracker.captureBegin(
      {kOtherChild, kOtherRoot, otherChildStream, otherRootStream});
  EXPECT_EQ(tracker.activeCaptureCount(), 5);

  EXPECT_EQ(tracker.captureEnd(kRoot), 0u);
  EXPECT_TRUE(tracker.hasActiveCaptures());
  EXPECT_EQ(tracker.activeCaptureCount(), 2);
  EXPECT_EQ(tracker.reuseStream(childStream), childStream);
  EXPECT_EQ(tracker.reuseStream(grandchildStream), grandchildStream);
  EXPECT_EQ(tracker.reuseStream(otherChildStream), otherRootStream);
  EXPECT_THAT(
      [&] { tracker.captureEnd(kChild); },
      throwsError("Capture is not registered or has already ended"));
  EXPECT_EQ(tracker.activeCaptureCount(), 2);
  EXPECT_EQ(tracker.captureEnd(kOtherChild), 0u);
  EXPECT_EQ(tracker.activeCaptureCount(), 1);
  EXPECT_EQ(tracker.captureEnd(kOtherRoot), 0u);
  EXPECT_EQ(tracker.activeCaptureCount(), 0);
}

TEST(
    CUDAGraphMemoryTrackerTest,
    InvalidRegistrationsDoNotChangeTheActiveCount) {
  graph::CaptureTracker tracker;
  constexpr c10::cuda::CaptureId_t kRoot = 1;
  constexpr c10::cuda::CaptureId_t kChild = 2;
  constexpr c10::cuda::CaptureId_t kMissingParent = 3;
  std::array<char, 2> storage{};
  const auto rootStream = reinterpret_cast<cudaStream_t>(&storage[0]);
  const auto childStream = reinterpret_cast<cudaStream_t>(&storage[1]);
  tracker.captureBegin({kRoot, std::nullopt, rootStream});
  EXPECT_THAT(
      [&] { tracker.captureBegin({0, std::nullopt, rootStream}); },
      throwsError("nonzero capture id"));
  EXPECT_THAT(
      [&] { tracker.captureBegin({kRoot, std::nullopt, rootStream}); },
      throwsError("Capture 1 is already registered"));
  EXPECT_THAT(
      [&] {
        tracker.captureBegin({kChild, kMissingParent, childStream, rootStream});
      },
      throwsError("Conditional capture parent is not active"));
  EXPECT_EQ(tracker.activeCaptureCount(), 1);
  EXPECT_EQ(tracker.reuseStream(childStream), childStream);
  EXPECT_THAT(
      [&] { tracker.captureEnd(kChild); },
      throwsError("Capture is not registered or has already ended"));
  EXPECT_EQ(tracker.activeCaptureCount(), 1);
  EXPECT_EQ(tracker.captureEnd(kRoot), 0u);
  EXPECT_FALSE(tracker.hasActiveCaptures());
}

TEST(CUDAGraphMemoryTrackerTest, NestedConditionalCapturesEndInnermostFirst) {
  graph::CaptureTracker tracker;
  constexpr c10::cuda::CaptureId_t kRoot = 1;
  constexpr c10::cuda::CaptureId_t kChild = 2;
  constexpr c10::cuda::CaptureId_t kGrandchild = 3;
  std::array<char, 3> storage{};
  const auto rootStream = reinterpret_cast<cudaStream_t>(&storage[0]);
  const auto childStream = reinterpret_cast<cudaStream_t>(&storage[1]);
  const auto grandchildStream = reinterpret_cast<cudaStream_t>(&storage[2]);
  tracker.captureBegin({kRoot, std::nullopt, rootStream});
  tracker.captureBegin({kChild, kRoot, childStream, rootStream});
  tracker.captureBegin({kGrandchild, kChild, grandchildStream, childStream});

  EXPECT_THAT(
      [&] { tracker.captureEnd(kChild); },
      throwsError(
          "Conditional capture 2 cannot end while a capture nested in it is "
          "active"));
  EXPECT_EQ(tracker.activeCaptureCount(), 3);
  EXPECT_EQ(tracker.reuseStream(childStream), rootStream);
  EXPECT_EQ(tracker.captureEnd(kGrandchild), 0u);
  EXPECT_EQ(tracker.captureEnd(kChild), 0u);
  EXPECT_EQ(tracker.captureEnd(kRoot), 0u);
  EXPECT_FALSE(tracker.hasActiveCaptures());
}

TEST(CUDAGraphMemoryTrackerTest, CountOnlyEndsCannotEndRegisteredCaptures) {
  graph::CaptureTracker tracker;
  constexpr c10::cuda::CaptureId_t kRoot = 1;
  std::array<char, 1> storage{};
  const auto rootStream = reinterpret_cast<cudaStream_t>(&storage[0]);
  tracker.captureBegin({kRoot, std::nullopt, rootStream});
  tracker.captureBegin();
  tracker.captureEnd();

  EXPECT_THAT(
      [&] { tracker.captureEnd(); },
      throwsError("captureEnd called with only registered captures active"));
  EXPECT_EQ(tracker.activeCaptureCount(), 1);
  EXPECT_EQ(tracker.captureEnd(kRoot), 0u);
  EXPECT_FALSE(tracker.hasActiveCaptures());
}
