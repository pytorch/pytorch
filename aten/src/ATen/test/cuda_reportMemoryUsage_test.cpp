#include <ATen/test/reportMemoryUsage.h>

#include <gtest/gtest.h>

#include <c10/cuda/CUDACachingAllocator.h>

TEST(DeviceCachingAllocator, check_reporter) {
  auto reporter = std::make_shared<TestMemoryReportingInfo>();
  c10::DebugInfoGuard guard(c10::DebugInfoKind::PROFILER_STATE, reporter);

  auto _200kb = 200 * 1024;
  auto _500mb = 500 * 1024 * 1024;

  auto allocator = c10::cuda::CUDACachingAllocator::get();

  auto alloc1 = allocator->allocate(_200kb);
  auto r = reporter->getLatestRecord();
  EXPECT_EQ(alloc1.get(), r.ptr);
  EXPECT_LE(_200kb, r.alloc_size);
  EXPECT_LE(_200kb, r.total_allocated);
  EXPECT_LE(_200kb, r.total_reserved);
  EXPECT_TRUE(r.device.is_cuda());

  auto alloc1_true_ptr = r.ptr;
  auto alloc1_true_alloc_size = r.alloc_size;

  // I bet pytorch will not waste that much memory
  EXPECT_LT(r.total_allocated, 2 * _200kb);
  // I bet pytorch will not reserve that much memory
  EXPECT_LT(r.total_reserved, _500mb);

  auto alloc2 = allocator->allocate(_500mb);
  r = reporter->getLatestRecord();
  EXPECT_EQ(alloc2.get(), r.ptr);
  EXPECT_LE(_500mb, r.alloc_size);
  EXPECT_LE(_200kb + _500mb, r.total_allocated);
  EXPECT_LE(_200kb + _500mb, r.total_reserved);
  EXPECT_TRUE(r.device.is_cuda());
  auto alloc2_true_ptr = r.ptr;
  auto alloc2_true_alloc_size = r.alloc_size;

  auto max_reserved = r.total_reserved;

  alloc1.clear();
  r = reporter->getLatestRecord();
  EXPECT_EQ(alloc1_true_ptr, r.ptr);
  EXPECT_EQ(-alloc1_true_alloc_size, r.alloc_size);
  EXPECT_EQ(alloc2_true_alloc_size, r.total_allocated);
  // alloc2 remain, it is a memory free operation, so it shouldn't reserve more
  // memory.
  EXPECT_TRUE(
      alloc2_true_alloc_size <= static_cast<int64_t>(r.total_reserved) &&
      r.total_reserved <= max_reserved);
  EXPECT_TRUE(r.device.is_cuda());

  alloc2.clear();
  r = reporter->getLatestRecord();
  EXPECT_EQ(alloc2_true_ptr, r.ptr);
  EXPECT_EQ(-alloc2_true_alloc_size, r.alloc_size);
  EXPECT_EQ(0, r.total_allocated);
  EXPECT_TRUE(r.total_reserved <= max_reserved);
  EXPECT_TRUE(r.device.is_cuda());
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  c10::cuda::CUDACachingAllocator::init(1);
  return RUN_ALL_TESTS();
}
