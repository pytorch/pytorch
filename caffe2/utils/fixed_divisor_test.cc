#include "caffe2/utils/fixed_divisor.h"
#include <gtest/gtest.h>

#include <random>

namespace caffe2 {

namespace {

void compareDivMod(int32_t v, int32_t divisor) {
  auto fixed = FixedDivisor<int32_t>(divisor);

  int nativeQ = v / divisor;
  int nativeR = v % divisor;

  int fixedQ = fixed.div(v);
  int fixedR = fixed.mod(v);

  EXPECT_EQ(fixedQ, nativeQ) << v << " / " << divisor
                             << " magic " << fixed.getMagic()
                             << " shift " << fixed.getShift()
                              << " quot " << fixedQ << " " << nativeQ;

  EXPECT_EQ(fixedR, nativeR) << v << " / " << divisor
                             << " magic " << fixed.getMagic()
                             << " shift " << fixed.getShift()
                             << " rem " << fixedR << " " << nativeR;
}

}

TEST(FixedDivisorTest, Test) {
  constexpr int32_t kMax = std::numeric_limits<int32_t>::max();

  // divide by 1
  compareDivMod(kMax, 1);
  compareDivMod(0, 1);
  compareDivMod(1, 1);

  // divide by max
  compareDivMod(kMax, kMax);
  compareDivMod(0, kMax);
  compareDivMod(1, kMax);

  // divide by random positive values
  std::random_device rd;
  std::uniform_int_distribution<int32_t> vDist(0, kMax);
  std::uniform_int_distribution<int32_t> qDist(1, kMax);

  std::uniform_int_distribution<int32_t> vSmallDist(0, 1000);
  std::uniform_int_distribution<int32_t> qSmallDist(1, 1000);
  for (int i = 0; i < 10000; ++i) {
    auto q = qDist(rd);
    auto v = vDist(rd);
    auto qSmall = qSmallDist(rd);
    auto vSmall = vSmallDist(rd);

    // random value
    compareDivMod(vSmall, qSmall);
    compareDivMod(vSmall, q);
    compareDivMod(v, qSmall);
    compareDivMod(v, q);

    // special values
    compareDivMod(kMax, qSmall);
    compareDivMod(0, qSmall);
    compareDivMod(1, qSmall);
    compareDivMod(kMax, q);
    compareDivMod(0, q);
    compareDivMod(1, q);

    compareDivMod(vSmall, 1);
    compareDivMod(vSmall, kMax);
    compareDivMod(v, 1);
    compareDivMod(v, kMax);
  }
}

}  // namespace caffe2
