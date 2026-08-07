#include <gtest/gtest.h>

#include <c10/util/generic_math.h>
#include <torch/headeronly/util/NumericUtils.h>
#include <cmath>
#include <limits>
namespace torch {
namespace aot_inductor {

TEST(TestMath, TestDivFloor) {
  EXPECT_EQ(c10::div_floor_floating(5., 0.), INFINITY);
  EXPECT_DOUBLE_EQ(c10::div_floor_floating(5., 2.), 2.);
  EXPECT_DOUBLE_EQ(c10::div_floor_floating(5., -2.), -3.);
  EXPECT_EQ(c10::div_floor_integer(5, 2), 2);
  EXPECT_EQ(c10::div_floor_integer(5, -2), -3);
}

TEST(TestMath, TestIsNan) {
  using torch::headeronly::_isnan;
  const float fnan = std::nanf("");

  EXPECT_FALSE(_isnan(1));
  EXPECT_FALSE(_isnan(1.0f));
  EXPECT_TRUE(_isnan(fnan));
  EXPECT_FALSE(_isnan(1.0));
  EXPECT_TRUE(_isnan(std::nan("")));

  EXPECT_FALSE(_isnan(torch::headeronly::complex<float>(1.0f, 2.0f)));
  EXPECT_TRUE(_isnan(torch::headeronly::complex<float>(fnan, 2.0f)));
  EXPECT_TRUE(_isnan(torch::headeronly::complex<float>(1.0f, fnan)));

  EXPECT_FALSE(_isnan(torch::headeronly::Half(1.0f)));
  EXPECT_TRUE(_isnan(torch::headeronly::Half(fnan)));
  EXPECT_FALSE(_isnan(torch::headeronly::BFloat16(1.0f)));
  EXPECT_TRUE(_isnan(torch::headeronly::BFloat16(fnan)));

  EXPECT_FALSE(_isnan(torch::headeronly::Float8_e5m2(1.0f)));
  EXPECT_TRUE(_isnan(torch::headeronly::Float8_e5m2(fnan)));
  EXPECT_TRUE(_isnan(torch::headeronly::Float8_e4m3fn(fnan)));
  EXPECT_TRUE(_isnan(torch::headeronly::Float8_e5m2fnuz(fnan)));
  EXPECT_TRUE(_isnan(torch::headeronly::Float8_e4m3fnuz(fnan)));
}

TEST(TestMath, TestIsInf) {
  using torch::headeronly::_isinf;
  const float finf = std::numeric_limits<float>::infinity();

  EXPECT_FALSE(_isinf(1));
  EXPECT_FALSE(_isinf(1.0f));
  EXPECT_TRUE(_isinf(finf));
  EXPECT_TRUE(_isinf(std::numeric_limits<double>::infinity()));

  EXPECT_FALSE(_isinf(torch::headeronly::Half(1.0f)));
  EXPECT_TRUE(_isinf(torch::headeronly::Half(finf)));
  EXPECT_FALSE(_isinf(torch::headeronly::BFloat16(1.0f)));
  EXPECT_TRUE(_isinf(torch::headeronly::BFloat16(finf)));
  EXPECT_TRUE(_isinf(torch::headeronly::Float8_e5m2(finf)));

  // the fn/fnuz formats have no encoding for infinity
  EXPECT_FALSE(_isinf(torch::headeronly::Float8_e4m3fn(1.0f)));
  EXPECT_FALSE(_isinf(torch::headeronly::Float8_e5m2fnuz(1.0f)));
  EXPECT_FALSE(_isinf(torch::headeronly::Float8_e4m3fnuz(1.0f)));
}

// On host these forward to the std functions; the fast-approximation
// intrinsics are only selected under __CUDA_ARCH__/__HIP_ARCH__/SYCL.
TEST(TestMath, TestExpLogTan) {
  using torch::headeronly::exp;
  using torch::headeronly::log;
  using torch::headeronly::log1p;
  using torch::headeronly::tan;

  EXPECT_FLOAT_EQ(exp(0.0f), 1.0f);
  EXPECT_FLOAT_EQ(exp(1.0f), std::exp(1.0f));
  EXPECT_FLOAT_EQ(log(1.0f), 0.0f);
  EXPECT_FLOAT_EQ(log(2.0f), std::log(2.0f));
  EXPECT_FLOAT_EQ(log1p(0.0f), 0.0f);
  EXPECT_FLOAT_EQ(log1p(1.0f), std::log1p(1.0f));
  EXPECT_FLOAT_EQ(tan(0.0f), 0.0f);
  EXPECT_FLOAT_EQ(tan(1.0f), std::tan(1.0f));

  // double is served by the explicit specializations, which skip the
  // "float or less precise type" static_assert on the primary template
  EXPECT_DOUBLE_EQ(exp(1.0), std::exp(1.0));
  EXPECT_DOUBLE_EQ(log(2.0), std::log(2.0));
  EXPECT_DOUBLE_EQ(log1p(1.0), std::log1p(1.0));
  EXPECT_DOUBLE_EQ(tan(1.0), std::tan(1.0));
}

} // namespace aot_inductor
} // namespace torch
