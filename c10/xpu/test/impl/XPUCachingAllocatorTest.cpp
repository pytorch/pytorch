#include <gtest/gtest.h>

#include <c10/util/irange.h>
#include <c10/xpu/XPUCachingAllocator.h>
#include <c10/xpu/XPUException.h>

bool has_xpu() {
  return c10::xpu::device_count() > 0;
}

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
  // We have reserved 500M of memory for resue. When allocating `ptr0` and
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

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  auto device = c10::xpu::device_count();
  if (device <= 0) {
    return 0;
  }
  c10::xpu::XPUCachingAllocator::init(device);
  return RUN_ALL_TESTS();
}
