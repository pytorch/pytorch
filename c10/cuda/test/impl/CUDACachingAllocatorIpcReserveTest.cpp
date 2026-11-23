#include <c10/core/AllocatorConfig.h>
#include <c10/cuda/CUDAAllocatorConfig.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/util/Exception.h>

#include <gtest/gtest.h>

#include <cstdlib>

// An IPC-imported expandable segment is fixed at the producer's handle count:
// map() is reachable only from map_block(), which runs on segments the
// allocator owns in expandable_segments_, and an imported segment is never
// inserted there. Reserving device-sized growth headroom for one therefore
// strands 9/8 of device memory worth of address space per import, which
// exhausts the 128 TiB user virtual address space after a few hundred of them.
// Lives in its own binary because it mutates the process-global allocator
// config and must set TORCH_CUDA_EXPANDABLE_SEGMENTS_IPC before the first
// allocation latches it.

namespace {

size_t deviceTotalBytes() {
  cudaDeviceProp prop{};
  C10_CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  return prop.totalGlobalMem;
}

} // namespace

TEST(ExpandableSegmentIpcReserveTest, ImportReservesOnlyWhatWasShared) {
  if (c10::cuda::device_count() == 0) {
    GTEST_SKIP() << "no CUDA device";
  }
  // Must precede the first allocation: the allocator reads this into a
  // function-local static and freezes each segment's handle type at map time.
  ASSERT_EQ(::setenv("TORCH_CUDA_EXPANDABLE_SEGMENTS_IPC", "1", 1), 0);
  c10::cuda::set_device(0);
  c10::CachingAllocator::setAllocatorSettings("expandable_segments:True");
  c10::cuda::CUDACachingAllocator::init(c10::cuda::device_count());
  ASSERT_TRUE(c10::cuda::CUDACachingAllocator::CUDAAllocatorConfig::
                  expandable_segments())
      << "expandable_segments must be enabled for this test to be meaningful";

  const size_t segment_size =
      c10::CachingAllocator::AcceleratorAllocatorConfig::large_segment_size();
  constexpr size_t kSharedSegments = 4;

  auto block = c10::cuda::CUDACachingAllocator::get()->allocate(
      kSharedSegments * segment_size);
  const auto shareable =
      c10::cuda::CUDACachingAllocator::shareIpcHandle(block.get());

  const size_t before =
      c10::cuda::CUDACachingAllocator::getExpandableSegmentsReservedBytes();
  const auto imported =
      c10::cuda::CUDACachingAllocator::getIpcDevPtr(shareable.handle);
  ASSERT_NE(imported, nullptr);
  const size_t reserved =
      c10::cuda::CUDACachingAllocator::getExpandableSegmentsReservedBytes() -
      before;

  // The import reserves the shared extent (plus at most one segment, if the
  // block straddles a boundary), never device-sized growth headroom.
  EXPECT_GE(reserved, kSharedSegments * segment_size);
  EXPECT_LE(reserved, (kSharedSegments + 1) * segment_size);
  EXPECT_LT(reserved, deviceTotalBytes())
      << "an imported segment cannot grow, so it must not reserve address "
         "space sized from device memory";
}
