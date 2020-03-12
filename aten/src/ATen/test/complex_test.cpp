#include <gtest/gtest.h>
#include "c10/util/Complex.h"

template<typename T, typename int_t>
static void TestBinaryOpForIntType(T real, T img, int_t num) {
  std::complex<T> c(real, img);
  ASSERT_EQ(c + num, std::complex<T>(real + num, img + num));
  ASSERT_EQ(num + c, std::complex<T>(num + real, num + img));
  ASSERT_EQ(c - num, std::complex<T>(real - num, num - num));
  ASSERT_EQ(num - c, std::complex<T>(num - real, num - img));
  ASSERT_EQ(c * num, std::complex<T>(real * num, img * num));
  ASSERT_EQ(num * c, std::complex<T>(num * real, num * img));
  ASSERT_EQ(c / num, std::complex<T>(real / num, img / num));
  ASSERT_EQ(num / c, std::complex<T>(num / real, num / img));
}

template<typename T>
static void TestBinaryOpForAllIntTypes(T real, T img, int8_t i) {
  TestBinaryOpForIntType<T, int8_t>(real, img, i, op);
  TestBinaryOpForIntType<T, int16_t>(real, img, i, op);
  TestBinaryOpForIntType<T, int32_t>(real, img, i, op);
  TestBinaryOpForIntType<T, int64_t>(real, img, i, op);
}

TEST(ComplexTest, Integer) {
  TestBinaryOpForAllIntTypes<float>(1.0, 0.1, 1);
  TestBinaryOpForAllIntTypes<double>(-1.3, -0.2, -2);
}
