#include <ATen/test/reportMemoryUsage.h>

#include <gtest/gtest.h>

#include <c10/core/CPUAllocator.h>

TEST(DefaultCPUAllocator, check_reporter) {
  auto reporter = std::make_shared<TestMemoryReportingInfo>();
  c10::DebugInfoGuard guard(c10::DebugInfoKind::PROFILER_STATE, reporter);

  auto allocator = c10::GetCPUAllocator();

  auto alloc1 = allocator->allocate(42);
  auto r = reporter->getLatestRecord();
  EXPECT_EQ(alloc1.get(), r.ptr);
  EXPECT_EQ(42, r.alloc_size);
  EXPECT_EQ(42, r.total_allocated);
  EXPECT_EQ(0, r.total_reserved);
  EXPECT_TRUE(r.device.is_cpu());

  auto alloc2 = allocator->allocate(1038);
  r = reporter->getLatestRecord();
  EXPECT_EQ(alloc2.get(), r.ptr);
  EXPECT_EQ(1038, r.alloc_size);
  EXPECT_EQ(1080, r.total_allocated);
  EXPECT_EQ(0, r.total_reserved);
  EXPECT_TRUE(r.device.is_cpu());

  auto alloc1_ptr = alloc1.get();
  alloc1.clear();
  r = reporter->getLatestRecord();
  EXPECT_EQ(alloc1_ptr, r.ptr);
  EXPECT_EQ(-42, r.alloc_size);
  EXPECT_EQ(1038, r.total_allocated);
  EXPECT_EQ(0, r.total_reserved);
  EXPECT_TRUE(r.device.is_cpu());

  auto alloc2_ptr = alloc2.get();
  alloc2.clear();
  r = reporter->getLatestRecord();
  EXPECT_EQ(alloc2_ptr, r.ptr);
  EXPECT_EQ(-1038, r.alloc_size);
  EXPECT_EQ(0, r.total_allocated);
  EXPECT_EQ(0, r.total_reserved);
  EXPECT_TRUE(r.device.is_cpu());
}
