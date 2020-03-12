#include <gtest/gtest.h>
#include "c10/util/Complex.h"

template<typename T, typename iT>
static void TestIntegerOp(T real, T img, iT i) {
  std::complex<T> c(real, img);
  ASSERT_EQ(c + i, std::complex<T>(real + i, img + i));
  ASSERT_EQ(i + c, std::complex<T>(i + real, i + img));
  ASSERT_EQ(c - i, std::complex<T>(real - i, img - i));
  ASSERT_EQ(i - c, std::complex<T>(i - real, i - img));
  ASSERT_EQ(c * i, std::complex<T>(real * i, img * i));
  ASSERT_EQ(i * c, std::complex<T>(i * real, i * img));
  ASSERT_EQ(c / i, std::complex<T>(real / i, img / i));
  ASSERT_EQ(i / c, std::complex<T>(i / real, i / img));
}

template<typename Op>
static void TestIntegerOpAllTypes(float real, float img, int8_t i) {
  TestIntegerOp<float, int8_t>(real, img, i, op);
  TestIntegerOp<double, int8_t>(real, img, i, op);
  TestIntegerOp<float, int16_t>(real, img, i, op);
  TestIntegerOp<double, int16_t>(real, img, i, op);
  TestIntegerOp<float, int32_t>(real, img, i, op);
  TestIntegerOp<double, int32_t>(real, img, i, op);
  TestIntegerOp<float, int64_t>(real, img, i, op);
  TestIntegerOp<double, int64_t>(real, img, i, op);
}

TEST(ComplexTest, Integer) {
  TestIntegerOpAllTypes(1.0, 0.1, 1);
  TestIntegerOpAllTypes(-1.3, -0.2, -2);
  TestIntegerOpAllTypes(1.1, -0.1, 3);
  TestIntegerOpAllTypes(-1.2, 10.1, -4);
}
