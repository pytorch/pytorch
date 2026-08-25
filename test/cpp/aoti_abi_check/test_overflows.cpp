#include <gtest/gtest.h>

#include <torch/headeronly/util/overflows.h>

#include <cmath>
#include <cstdint>
#include <limits>

namespace torch {
namespace aot_inductor {

TEST(TestOverflows, TestIntegral) {
  using torch::headeronly::overflows;

  EXPECT_TRUE(overflows<int8_t>(200));
  EXPECT_FALSE(overflows<int8_t>(100));
  EXPECT_TRUE(overflows<uint8_t>(-1));
}

TEST(TestOverflows, TestFloatToWideIntBoundary) {
  using torch::headeronly::overflows;

  // 2^63 is exactly representable in double and is INT64_MAX+1.
  EXPECT_TRUE(overflows<int64_t>(9223372036854775808.0));
  EXPECT_FALSE(overflows<int64_t>(9223372036854774784.0));
}

TEST(TestOverflows, TestNan) {
  using torch::headeronly::overflows;

  EXPECT_TRUE(overflows<int32_t>(std::nan("")));
}

TEST(TestOverflows, TestComplex) {
  using torch::headeronly::complex;
  using torch::headeronly::overflows;

  EXPECT_FALSE(
      (overflows<complex<float>, complex<double>>(complex<double>(1.0, 2.0))));
}

} // namespace aot_inductor
} // namespace torch
