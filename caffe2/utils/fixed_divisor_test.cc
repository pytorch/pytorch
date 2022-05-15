#include "caffe2/utils/fixed_divisor.h"

#include <gtest/gtest.h>

#include <random>

namespace caffe2 {

namespace {

void CompareDivMod(int32_t v, int32_t divisor) {
  auto fixed = FixedDivisor<int32_t>(divisor);

  int native_q = v / divisor;
  int native_r = v % divisor;

  int fixed_q = fixed.Div(v);
  int fixed_r = fixed.Mod(v);

#if !defined(USE_ROCM)
  EXPECT_EQ(native_q, fixed_q)
      << v << " / " << divisor << " magic " << fixed.magic() << " shift "
      << fixed.shift() << " quot " << fixed_q << " " << native_q;

  EXPECT_EQ(native_r, fixed_r)
      << v << " / " << divisor << " magic " << fixed.magic() << " shift "
      << fixed.shift() << " rem " << fixed_r << " " << native_r;
#endif
}

} // namespace

TEST(FixedDivisorTest, FixedDivisorInt32Test) {
  constexpr int32_t kMax = std::numeric_limits<int32_t>::max();

  // divide by 1
  CompareDivMod(kMax, 1);
  CompareDivMod(0, 1);
  CompareDivMod(1, 1);

  // divide by max
  CompareDivMod(kMax, kMax);
  CompareDivMod(0, kMax);
  CompareDivMod(1, kMax);

  // divide by random positive values
  std::random_device rd;
  std::uniform_int_distribution<int32_t> v_dist(0, kMax);
  std::uniform_int_distribution<int32_t> q_dist(1, kMax);

  std::uniform_int_distribution<int32_t> v_small_dist(0, 1000);
  std::uniform_int_distribution<int32_t> q_small_dist(1, 1000);
  for (int i = 0; i < 10000; ++i) {
    auto q = q_dist(rd);
    auto v = v_dist(rd);
    auto q_small = q_small_dist(rd);
    auto v_small = v_small_dist(rd);

    // random value
    CompareDivMod(v_small, q_small);
    CompareDivMod(v_small, q);
    CompareDivMod(v, q_small);
    CompareDivMod(v, q);

    // special values
    CompareDivMod(kMax, q_small);
    CompareDivMod(0, q_small);
    CompareDivMod(1, q_small);
    CompareDivMod(kMax, q);
    CompareDivMod(0, q);
    CompareDivMod(1, q);

    CompareDivMod(v_small, 1);
    CompareDivMod(v_small, kMax);
    CompareDivMod(v, 1);
    CompareDivMod(v, kMax);
  }
}

} // namespace caffe2
