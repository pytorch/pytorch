#include <gtest/gtest.h>

#include <c10/core/AllocatorConfig.h>
#include <c10/xpu/XPUCachingAllocator.h>
#include <c10/xpu/test/impl/XPUTest.h>

// Param: whether expandable segments are enabled.
class XPUAllocatorIPCTest : public ::testing::TestWithParam<bool> {
 protected:
  void SetUp() override {
    if (GetParam()) {
#if SYCL_COMPILER_VERSION >= 20260200
      c10::CachingAllocator::setAllocatorSettings("expandable_segments:True");
#else
      GTEST_SKIP()
          << "Expandable segment IPC sharing requires SYCL compiler 2026.2 or later.";
#endif
    }
  }

  void TearDown() override {
    if (GetParam()) {
      c10::CachingAllocator::setAllocatorSettings("expandable_segments:False");
    }
    c10::xpu::XPUCachingAllocator::emptyCache();
  }
};

TEST_P(XPUAllocatorIPCTest, ShareAndGetRoundTrip) {
  auto* allocator = c10::xpu::XPUCachingAllocator::get();
  const auto _4mb = 4 * 1024 * 1024;
  auto buf = allocator->allocate(_4mb);
  void* ptr = buf.get();

  auto sh = c10::xpu::XPUCachingAllocator::shareIpcHandle(ptr);
  EXPECT_EQ(sh.offset, 0);
  EXPECT_FALSE(sh.handle.empty());

  auto devptr = c10::xpu::XPUCachingAllocator::getIpcDevPtr(sh.handle);
  ASSERT_NE(devptr, nullptr);
}

TEST_P(XPUAllocatorIPCTest, ShareOffsetSubAllocation) {
  auto* allocator = c10::xpu::XPUCachingAllocator::get();
  // 20M memory is reserved, can be reused later.
  {
    auto _20mb = 20 * 1024 * 1024;
    auto cache = allocator->allocate(_20mb);
  }
  const auto _10mb = 10 * 1024 * 1024;
  auto base_buf = allocator->allocate(_10mb);

  const auto _5mb = 5 * 1024 * 1024;
  auto sub_buf = allocator->allocate(_5mb);

  auto sh = c10::xpu::XPUCachingAllocator::shareIpcHandle(sub_buf.get());
  EXPECT_GT(sh.offset, 0);
  EXPECT_FALSE(sh.handle.empty());

  auto devptr = c10::xpu::XPUCachingAllocator::getIpcDevPtr(sh.handle);
  ASSERT_NE(devptr, nullptr);
}

TEST_P(XPUAllocatorIPCTest, GetIpcDevPtrCacheHit) {
  auto* allocator = c10::xpu::XPUCachingAllocator::get();
  const auto _4mb = 4 * 1024 * 1024;
  auto buf = allocator->allocate(_4mb);

  auto sh = c10::xpu::XPUCachingAllocator::shareIpcHandle(buf.get());

  auto sp1 = c10::xpu::XPUCachingAllocator::getIpcDevPtr(sh.handle);
  auto sp2 = c10::xpu::XPUCachingAllocator::getIpcDevPtr(sh.handle);

  // Both calls must return shared_ptrs sharing the same control block.
  EXPECT_EQ(sp1.get(), sp2.get());
  EXPECT_EQ(sp1.use_count(), 2);
}

TEST_P(XPUAllocatorIPCTest, GetIpcDevPtrCacheEviction) {
  auto* allocator = c10::xpu::XPUCachingAllocator::get();
  const auto _4mb = 4 * 1024 * 1024;
  auto buf = allocator->allocate(_4mb);

  auto sh = c10::xpu::XPUCachingAllocator::shareIpcHandle(buf.get());

  {
    auto sp = c10::xpu::XPUCachingAllocator::getIpcDevPtr(sh.handle);
    EXPECT_EQ(sp.use_count(), 1);
  }
  // sp is gone; the deleter erased the cache entry. A fresh call re-opens the
  // IPC mapping with a new control block.
  auto sp2 = c10::xpu::XPUCachingAllocator::getIpcDevPtr(sh.handle);
  ASSERT_NE(sp2, nullptr);
  EXPECT_EQ(sp2.use_count(), 1);
}

TEST_P(XPUAllocatorIPCTest, SameProcessRoundTrip) {
  const auto _4mb = 4 * 1024 * 1024;
  const int numel = _4mb / sizeof(int);

  auto* allocator = c10::xpu::XPUCachingAllocator::get();
  auto buf = allocator->allocate(_4mb);
  void* dev_ptr = buf.get();

  std::vector<int> hostData(numel);
  initHostData(hostData.data(), numel);

  auto sh = c10::xpu::XPUCachingAllocator::shareIpcHandle(dev_ptr);
  auto dev_ptr_ipc = c10::xpu::XPUCachingAllocator::getIpcDevPtr(sh.handle);
  void* ipc_addr = static_cast<char*>(dev_ptr_ipc.get()) + sh.offset;

  c10::xpu::getCurrentXPUStream()
      .queue()
      .memcpy(ipc_addr, hostData.data(), _4mb)
      .wait();
  clearHostData(hostData.data(), numel);
  c10::xpu::getCurrentXPUStream()
      .queue()
      .memcpy(hostData.data(), dev_ptr, _4mb)
      .wait();

  validateHostData(hostData.data(), numel);
}

// With expandable segments, the first allocation pins segment 0, so the shared
// block starts at a non-zero segment index and spans several segments.
TEST_P(XPUAllocatorIPCTest, SameProcessRoundTripMultiSegment) {
  auto* allocator = c10::xpu::XPUCachingAllocator::get();
  const auto seg_size =
      c10::CachingAllocator::AcceleratorAllocatorConfig::large_segment_size();
  auto head = allocator->allocate(seg_size);
  const size_t size = 2 * seg_size + seg_size / 2;
  const int numel = static_cast<int>(size / sizeof(int));
  auto buf = allocator->allocate(size);
  void* dev_ptr = buf.get();

  std::vector<int> hostData(numel);
  initHostData(hostData.data(), numel);

  auto sh = c10::xpu::XPUCachingAllocator::shareIpcHandle(dev_ptr);
  auto dev_ptr_ipc = c10::xpu::XPUCachingAllocator::getIpcDevPtr(sh.handle);
  void* ipc_addr = static_cast<char*>(dev_ptr_ipc.get()) + sh.offset;

  c10::xpu::getCurrentXPUStream()
      .queue()
      .memcpy(ipc_addr, hostData.data(), size)
      .wait();
  clearHostData(hostData.data(), numel);
  c10::xpu::getCurrentXPUStream()
      .queue()
      .memcpy(hostData.data(), dev_ptr, size)
      .wait();

  validateHostData(hostData.data(), numel);
}

INSTANTIATE_TEST_SUITE_P(
    AllocatorMode,
    XPUAllocatorIPCTest,
    ::testing::Bool(),
    [](const ::testing::TestParamInfo<bool>& info) {
      return std::string(info.param ? "ExpandableSegment" : "Regular");
    });

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  auto device = c10::xpu::device_count();
  if (device <= 0) {
    return 0;
  }
  if (!c10::xpu::get_raw_device(0).has(sycl::aspect::ext_oneapi_ipc_memory)) {
    return 0;
  }
  c10::xpu::XPUCachingAllocator::init(device);
  return RUN_ALL_TESTS();
}
