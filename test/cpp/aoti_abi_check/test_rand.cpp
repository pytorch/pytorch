#include <gtest/gtest.h>

#include <ATen/core/PhiloxRNGEngine.h>

#include <cstdint>
#include <iostream>
namespace torch {
namespace aot_inductor {

int64_t randint64_cpu(
    uint32_t seed,
    uint32_t offset,
    int64_t low,
    int64_t high) {
  auto gen = at::Philox4_32(seed, 0, offset);
  uint64_t r0 = gen();
  uint64_t r1 = gen();
  uint64_t result = r0 | (r1 << 32);
  return static_cast<int64_t>(result % (high - low)) + low;
}

TEST(TestRand, TestRandn) {
  at::Philox4_32 engine_1(1, 0, 0);
  float a = engine_1.randn(10);
  at::Philox4_32 engine_2(1, 0, 0);
  float b = engine_2.randn(10);

  EXPECT_EQ(a, b);
}

TEST(TestRand, TestRandint64) {
  int64_t a = randint64_cpu(0xffffffff, 100, 0, INT64_MAX);
  int64_t b = randint64_cpu(0xffffffff, 100, 0, INT64_MAX);

  EXPECT_EQ(a, b);
}

} // namespace aot_inductor
} // namespace torch
