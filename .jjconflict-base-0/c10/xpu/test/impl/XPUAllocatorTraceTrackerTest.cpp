#include <gtest/gtest.h>

#include <c10/xpu/XPUCachingAllocator.h>

static int segmentAllocCalled = 0;
static int segmentFreeCalled = 0;

static void SegmentAllocTraceTracker(
    const c10::CachingDeviceAllocator::TraceEntry& te) {
  if (te.action_ ==
      c10::CachingDeviceAllocator::TraceEntry::Action::SEGMENT_ALLOC) {
    segmentAllocCalled++;
  }
}

static void SegmentFreeTraceTracker(
    const c10::CachingDeviceAllocator::TraceEntry& te) {
  if (te.action_ ==
      c10::CachingDeviceAllocator::TraceEntry::Action::SEGMENT_FREE) {
    segmentFreeCalled++;
  }
}

static void allocateLargeBuffer() {
  const auto _500mb = 500 * 1024 * 1024;
  auto* allocator = c10::xpu::XPUCachingAllocator::get();
  auto buffer = allocator->allocate(_500mb);
}

TEST(AllocatorTraceTracker, TrackMallocFree) {
  c10::xpu::XPUCachingAllocator::attachAllocatorTraceTracker(
      &SegmentAllocTraceTracker);
  c10::xpu::XPUCachingAllocator::attachAllocatorTraceTracker(
      &SegmentFreeTraceTracker);

  // Expect to trigger segment allocation for large buffer
  // and expect the buffer would be marked as inactive when return from
  // allocateLargeBuffer and be freed when calling emptyCache
  allocateLargeBuffer();
  ASSERT_EQ(segmentAllocCalled, 1);

  // Expect allocated buffer has been released back to allocator, thus empty
  // cache would trigger segment free
  c10::xpu::XPUCachingAllocator::emptyCache();
  ASSERT_EQ(segmentFreeCalled, 1);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  auto device = c10::xpu::device_count();
  if (device <= 0) {
    return 0;
  }
  c10::xpu::XPUCachingAllocator::init(device);
  return RUN_ALL_TESTS();
}
