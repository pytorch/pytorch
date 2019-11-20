#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/MemoryOverlap.h>

using namespace at;

void TestContiguousTensor(DeprecatedTypeProperties& T) {
  auto a = randn({2, 3}, T);
  auto b = randn({3}, T);
  auto c = randn({2, 1, 5}, T);
  auto d = randn({10, 2, 5, 5}, T);
  auto e = randn({1, 2, 5, 1}, T);

  ASSERT_TRUE(has_internal_overlap(a) == MemOverlap::NO);
  ASSERT_TRUE(has_internal_overlap(b) == MemOverlap::NO);
  ASSERT_TRUE(has_internal_overlap(c) == MemOverlap::NO);
  ASSERT_TRUE(has_internal_overlap(d) == MemOverlap::NO);
  ASSERT_TRUE(has_internal_overlap(e) == MemOverlap::NO);
}

void TestOverlapTensor(DeprecatedTypeProperties& T) {
  auto a = randn({10, 1, 10}, T).expand({10, 10, 10});
  auto b = randn({1, 2}, T).expand({10, 2});
  auto c = randn({4, 1}, T).expand({4, 4});
  auto d = randn({2, 1, 4, 1}, T).expand({2, 4, 4, 1});

  ASSERT_TRUE(has_internal_overlap(a) == MemOverlap::YES);
  ASSERT_TRUE(has_internal_overlap(b) == MemOverlap::YES);
  ASSERT_TRUE(has_internal_overlap(c) == MemOverlap::YES);
  ASSERT_TRUE(has_internal_overlap(d) == MemOverlap::YES);

  /* hard case where there's overlap*/
  auto e = randn({16}, T);
  e.set_(e.storage(), e.storage_offset(), {2, 4, 2, 2}, {8, 2, 2, 1});
  ASSERT_TRUE(has_internal_overlap(e) != MemOverlap::NO);
}

void TestNonOverlapTensor(DeprecatedTypeProperties& T) {

  /* easy non-packed tensor */
  auto a = randn({10, 4, 10}, T).slice(2, 1, 3);
  ASSERT_TRUE(has_internal_overlap(a) == MemOverlap::NO);
  /* easy size 1 dimension with strange stride */
  auto b = randn({3, 1, 5}, T);
  ASSERT_TRUE(has_internal_overlap(b) == MemOverlap::NO);

  /* hard case where there's no overlap*/
  auto c = randn({10}, T);
  c.set_(c.storage(), c.storage_offset(), {2, 3}, {4, 3});
  ASSERT_TRUE(has_internal_overlap(c) != MemOverlap::YES);
}

TEST(MemoryOverlapTest, MemoryOverlap) {
  manual_seed(123);
  DeprecatedTypeProperties& T = CPU(kFloat);

  TestContiguousTensor(T);
  TestOverlapTensor(T);
  TestNonOverlapTensor(T);
}
