#include <c10/cuda/CUDACachingAllocator.h>
#include <gtest/gtest.h>

static int segmentAllocCalled = 0;
static int segmentFreeCalled = 0;

static void SegmentAllocTraceTracker(
    const c10::cuda::CUDACachingAllocator::TraceEntry& te) {
  if (te.action_ ==
      c10::cuda::CUDACachingAllocator::TraceEntry::Action::SEGMENT_ALLOC) {
    segmentAllocCalled++;
  }
}

static void SegmentFreeTraceTracker(
    const c10::cuda::CUDACachingAllocator::TraceEntry& te) {
  if (te.action_ ==
      c10::cuda::CUDACachingAllocator::TraceEntry::Action::SEGMENT_FREE) {
    segmentFreeCalled++;
  }
}

static void allocateLargeBuffer() {
  const auto _500mb = 500 * 1024 * 1024;
  auto* allocator = c10::cuda::CUDACachingAllocator::get();
  auto buffer = allocator->allocate(_500mb);
}

TEST(AllocatorTraceTracker, TrackMallocFree) {
  c10::cuda::CUDACachingAllocator::attachAllocatorTraceTracker(
      &SegmentAllocTraceTracker);
  c10::cuda::CUDACachingAllocator::attachAllocatorTraceTracker(
      &SegmentFreeTraceTracker);

  // Expect to trigger segment allocation for large buffer
  // and expect the buffer would be marked as inactive when return from
  // allocateLargeBuffer and be freed when calling emptyCache
  allocateLargeBuffer();
  ASSERT_EQ(segmentAllocCalled, 1);

  // Expect allocated buffer has been released back to allocator, thus empty
  // cache would trigger segment free
  c10::cuda::CUDACachingAllocator::emptyCache();
  ASSERT_EQ(segmentFreeCalled, 1);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  c10::cuda::CUDACachingAllocator::init(1);
  return RUN_ALL_TESTS();
}
