#include <gtest/gtest.h>

#include <c10/util/irange.h>
#include <c10/xpu/XPUCachingAllocator.h>
#include <c10/xpu/XPUException.h>

TEST(XPUCachingAllocatorTest, GetXPUAllocator) {
  auto* allocator = c10::xpu::XPUCachingAllocator::get();

  auto _500mb = 500 * 1024 * 1024;
  auto buffer = allocator->allocate(_500mb);
  EXPECT_TRUE(buffer.get());

  auto* xpu_allocator = c10::GetAllocator(buffer.device().type());
  EXPECT_EQ(allocator, xpu_allocator);
}

TEST(XPUCachingAllocatorTest, DeviceCachingAllocate) {
  c10::xpu::XPUCachingAllocator::emptyCache();
  auto* allocator = c10::xpu::XPUCachingAllocator::get();
  // 500M memory is reserved, can be reused later.
  {
    auto _500mb = 500 * 1024 * 1024;
    auto cache = allocator->allocate(_500mb);
  }
  auto _10mb = 10 * 1024 * 1024;
  auto buffer = allocator->allocate(_10mb);
  void* ptr0 = buffer.get();
  // tmp is not allocated via device caching allocator.
  void* tmp = sycl::aligned_alloc_device(
      512, _10mb, c10::xpu::get_raw_device(0), c10::xpu::get_device_context());
  void* ptr1 = c10::xpu::XPUCachingAllocator::raw_alloc(_10mb);
  // We have reserved 500M memory that can be reused. When we allocate ptr0
  // and ptr1 via device caching allocator, they should be on the same block.
  // And ptr1 is the next block of ptr0, like [ptr0, ptr1]. This is because tmp
  // pointer is not allocated via device caching allocator so that it can NOT
  // reuse our reserved memory. So the offset between ptr0 and ptr1 should equal
  // to ptr0's size (10M).
  auto diff = static_cast<char*>(ptr1) - static_cast<char*>(ptr0);
  EXPECT_EQ(diff, _10mb);
  c10::xpu::XPUCachingAllocator::raw_delete(ptr1);
  sycl::free(tmp, c10::xpu::get_device_context());
  c10::xpu::XPUCachingAllocator::emptyCache();
}

TEST(XPUCachingAllocatorTest, AllocateMemory) {
  c10::xpu::XPUCachingAllocator::emptyCache();
  auto* allocator = c10::xpu::XPUCachingAllocator::get();
  auto _10mb = 10 * 1024 * 1024;
  auto buffer = allocator->allocate(_10mb);
  auto* deviceData = static_cast<int*>(buffer.get());

  constexpr int numel = 1024;
  int hostData[numel];
  for (const auto i : c10::irange(numel)) {
    hostData[i] = i;
  }

  auto stream = c10::xpu::getStreamFromPool();
  // H2D
  stream.queue().memcpy(deviceData, hostData, sizeof(int) * numel);
  c10::xpu::syncStreamsOnDevice();

  for (const auto i : c10::irange(numel)) {
    hostData[i] = 0;
  }

  // D2H
  stream.queue().memcpy(hostData, deviceData, sizeof(int) * numel);
  c10::xpu::syncStreamsOnDevice();

  for (const auto i : c10::irange(numel)) {
    EXPECT_EQ(hostData[i], i);
  }
  c10::xpu::XPUCachingAllocator::emptyCache();
}

TEST(XPUCachingAllocatorTest, DeviceCachingAllocateByExternalStream) {
  c10::xpu::XPUCachingAllocator::emptyCache();
  auto* allocator = c10::xpu::XPUCachingAllocator::get();
  sycl::queue* ext_queue = new sycl::queue(
      c10::xpu::get_device_context(),
      c10::xpu::get_raw_device(0),
      c10::xpu::asyncHandler,
      {sycl::property::queue::in_order()});
  // 500M memory is reserved, can be reused later.
  {
    c10::xpu::XPUStream ext_stream =
        c10::xpu::getStreamFromExternal(ext_queue, 0);
    c10::xpu::setCurrentXPUStream(ext_stream);
    auto _500mb = 500 * 1024 * 1024;
    auto cache = allocator->allocate(_500mb);
  }
  auto _10mb = 10 * 1024 * 1024;
  auto buffer = allocator->allocate(_10mb);
  void* ptr0 = buffer.get();
  // tmp is not allocated via device caching allocator.
  void* tmp = sycl::aligned_alloc_device(
      512, _10mb, c10::xpu::get_raw_device(0), c10::xpu::get_device_context());
  void* ptr1 = c10::xpu::XPUCachingAllocator::raw_alloc(_10mb);
  // We have reserved 500M of memory for reuse. When allocating `ptr0` and
  // `ptr1` through the device caching allocator, they should be allocated from
  // the same block. Specifically, `ptr1` should follow immediately after `ptr0`
  // in the block, forming a sequence like [ptr0, ptr1]. This behavior occurs
  // because the `tmp` pointer is not allocated through the device caching
  // allocator, meaning it cannot reuse the reserved memory. As a result, the
  // offset between `ptr0` and `ptr1` should match the size of `ptr0` (10M in
  // this case).
  auto diff = static_cast<char*>(ptr1) - static_cast<char*>(ptr0);
  EXPECT_EQ(diff, _10mb);
  c10::xpu::XPUCachingAllocator::raw_delete(ptr1);
  sycl::free(tmp, c10::xpu::get_device_context());
  delete ext_queue;
  c10::xpu::XPUCachingAllocator::emptyCache();
}

TEST(XPUCachingAllocatorTest, NoSplitPool) {
  c10::xpu::XPUCachingAllocator::emptyCache();

  const auto device = c10::DeviceIndex(0);
  sycl::queue* stream_queue = &c10::xpu::getCurrentXPUStream(device).queue();
  auto filter = [stream_queue](sycl::queue* q) { return q == stream_queue; };

  // Use a fixed user-created pool id that won't collide with any MemPool
  // created by other tests.
  const c10::MempoolId_t pool_id{0, 9999};

  // Step 1: pre-fill the pool cache with a 20 MB block.
  const size_t _20mb = 20 * 1024 * 1024;
  c10::xpu::XPUCachingAllocator::beginAllocateToPool(device, pool_id, filter);
  void* large_ptr = c10::xpu::XPUCachingAllocator::raw_alloc(_20mb);
  c10::xpu::XPUCachingAllocator::raw_delete(large_ptr);

  // Step 2: mark pool as no-split.
  c10::xpu::XPUCachingAllocator::setNoSplit(device, pool_id);

  // Step 3: allocate a smaller block (10 MB); the 20 MB cached block must not
  // be split.
  const size_t _10mb = 10 * 1024 * 1024;
  void* small_ptr = c10::xpu::XPUCachingAllocator::raw_alloc(_10mb);
  c10::xpu::XPUCachingAllocator::endAllocateToPool(device, pool_id);

  // Step 4: inspect pool snapshot.
  auto snap = c10::xpu::XPUCachingAllocator::snapshot(pool_id);
  bool found = false;
  for (const auto& seg : snap.segments) {
    for (const auto& blk : seg.blocks) {
      if (blk.allocated) {
        // Physical size unchanged (no split); only requested_size is smaller.
        EXPECT_GE(blk.size, _20mb);
        EXPECT_EQ(blk.requested_size, _10mb);
        found = true;
      }
    }
  }
  EXPECT_TRUE(found) << "No allocated block found in pool snapshot";

  c10::xpu::XPUCachingAllocator::raw_delete(small_ptr);
  c10::xpu::XPUCachingAllocator::emptyCache(pool_id);
  c10::xpu::XPUCachingAllocator::releasePool(device, pool_id);
}

TEST(XPUCachingAllocatorTest, UseOnOOMPool) {
  c10::xpu::XPUCachingAllocator::emptyCache();

  const auto device = c10::DeviceIndex(0);
  sycl::queue* stream_queue = &c10::xpu::getCurrentXPUStream(device).queue();
  auto filter = [stream_queue](sycl::queue* q) { return q == stream_queue; };
  auto orig_fraction = c10::xpu::XPUCachingAllocator::getMemoryFraction(device);

  // Use a fixed user-created pool id that won't collide with any MemPool
  // created by other tests.
  const c10::MempoolId_t pool_id{0, 99999};

  // Step 1: pre-fill the OOM pool cache with a 20 MB block.
  const size_t _20mb = 20 * 1024 * 1024;
  c10::xpu::XPUCachingAllocator::beginAllocateToPool(device, pool_id, filter);
  void* oom_ptr = c10::xpu::XPUCachingAllocator::raw_alloc(_20mb);
  c10::xpu::XPUCachingAllocator::raw_delete(oom_ptr);

  // Step 2: register the pool as the OOM fallback.
  c10::xpu::XPUCachingAllocator::setUseOnOOM(device, pool_id, true);

  // Step 3: set a near-zero memory fraction so alloc_block always fails for new
  // device allocations, forcing the allocator into try_mempool_fallback.
  c10::xpu::XPUCachingAllocator::setMemoryFraction(1e-9, device);
  c10::xpu::XPUCachingAllocator::endAllocateToPool(device, pool_id);

  // Step 4: so this must be served by the OOM pool's cached 20 MB block.
  const size_t _10mb = 10 * 1024 * 1024;
  void* ptr = c10::xpu::XPUCachingAllocator::raw_alloc(_10mb);
  EXPECT_NE(ptr, nullptr)
      << "Allocation should have succeeded via OOM pool fallback";
  ASSERT_EQ(oom_ptr, ptr)
      << "Allocation should have returned the pre-cached OOM pool block";

  // Step 5: restore a permissive fraction before cleanup.
  c10::xpu::XPUCachingAllocator::setMemoryFraction(orig_fraction, device);

  c10::xpu::XPUCachingAllocator::raw_delete(ptr);
  c10::xpu::XPUCachingAllocator::emptyCache();
  c10::xpu::XPUCachingAllocator::setUseOnOOM(device, pool_id, false);
  c10::xpu::XPUCachingAllocator::releasePool(device, pool_id);
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
