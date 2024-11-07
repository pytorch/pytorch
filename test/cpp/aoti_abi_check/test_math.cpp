#include <gtest/gtest.h>

#include <ATen/NumericUtils.h>
#include <c10/util/generic_math.h>
#include <cmath>
namespace torch {
namespace aot_inductor {

TEST(TestMath, TestDivFloor) {
  EXPECT_EQ(c10::div_floor_floating(5., 0.), INFINITY);
  EXPECT_DOUBLE_EQ(c10::div_floor_floating(5., 2.), 2.);
  EXPECT_DOUBLE_EQ(c10::div_floor_floating(5., -2.), -3.);
  EXPECT_EQ(c10::div_floor_integer(5, 2), 2);
  EXPECT_EQ(c10::div_floor_integer(5, -2), -3);
}

TEST(TestMath, TestNan) {
  EXPECT_FALSE(at::_isnan(1.0));
  EXPECT_TRUE(at::_isnan(std::nan("")));
}

} // namespace aot_inductor
} // namespace torch
