#include <gtest/gtest.h>
#include "c10/util/Complex.h"

template<typename T, typename int_t>
static void TestBinaryOpsForIntType(T real, T img, int_t num) {
  std::complex<T> c(real, img);
  ASSERT_EQ(c + num, std::complex<T>(real + num, img));
  ASSERT_EQ(num + c, std::complex<T>(num + real, img));
  ASSERT_EQ(c - num, std::complex<T>(real - num, img));
  ASSERT_EQ(num - c, std::complex<T>(num - real, -img));
  ASSERT_EQ(c * num, std::complex<T>(real * num, img * num));
  ASSERT_EQ(num * c, std::complex<T>(num * real, num * img));
  ASSERT_EQ(c / num, std::complex<T>(real / num, img / num));
  T r2 = real * real + img * img;
  ASSERT_EQ(num / c, std::complex<T>(num * real / r2, - num * img / r2));
}

template<typename T>
static void TestBinaryOpsForAllIntTypes(T real, T img, int8_t i) {
  TestBinaryOpsForIntType<T, int8_t>(real, img, i);
  TestBinaryOpsForIntType<T, int16_t>(real, img, i);
  TestBinaryOpsForIntType<T, int32_t>(real, img, i);
  TestBinaryOpsForIntType<T, int64_t>(real, img, i);
}

TEST(ComplexTest, Integer) {
  TestBinaryOpsForAllIntTypes<float>(1.0, 0.1, 1);
  TestBinaryOpsForAllIntTypes<double>(-1.3, -0.2, -2);
}
