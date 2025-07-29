#include <gtest/gtest.h>

#include <c10/util/BFloat16-math.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Float8_e4m3fn.h>
#include <c10/util/Float8_e4m3fnuz.h>
#include <c10/util/Float8_e5m2.h>
#include <c10/util/Float8_e5m2fnuz.h>
#include <c10/util/Half.h>
#include <c10/util/complex.h>

#include <torch/headeronly/util/bits.h>
#include <torch/headeronly/util/quint8.h>
#include <torch/headeronly/util/quint4x2.h>
#include <torch/headeronly/util/quint2x4.h>
#include <torch/headeronly/util/qint8.h>
#include <torch/headeronly/util/qint32.h>

TEST(TestDtype, TestBFloat16) {
  c10::BFloat16 a = 1.0f;
  c10::BFloat16 b = 2.0f;
  c10::BFloat16 add = 3.0f;
  c10::BFloat16 sub = -1.0f;
  c10::BFloat16 mul = 2.0f;
  c10::BFloat16 div = 0.5f;

  EXPECT_EQ(a + b, add);
  EXPECT_EQ(a - b, sub);
  EXPECT_EQ(a * b, mul);
  EXPECT_EQ(a / b, div);
}

TEST(TestDtype, TestFloat8_e4m3fn) {
  c10::Float8_e4m3fn a = 1.0f;
  c10::Float8_e4m3fn b = 2.0f;
  c10::Float8_e4m3fn add = 3.0f;
  c10::Float8_e4m3fn sub = -1.0f;
  c10::Float8_e4m3fn mul = 2.0f;
  c10::Float8_e4m3fn div = 0.5f;

  EXPECT_EQ(a + b, add);
  EXPECT_EQ(a - b, sub);
  EXPECT_EQ(a * b, mul);
  EXPECT_EQ(a / b, div);
}

TEST(TestDtype, TestFloat8_e4m3fuz) {
  c10::Float8_e4m3fnuz a = 1.0f;
  c10::Float8_e4m3fnuz b = 2.0f;
  c10::Float8_e4m3fnuz add = 3.0f;
  c10::Float8_e4m3fnuz sub = -1.0f;
  c10::Float8_e4m3fnuz mul = 2.0f;
  c10::Float8_e4m3fnuz div = 0.5f;

  EXPECT_EQ(a + b, add);
  EXPECT_EQ(a - b, sub);
  EXPECT_EQ(a * b, mul);
  EXPECT_EQ(a / b, div);
}

TEST(TestDtype, TestFloat8_e5m2) {
  c10::Float8_e5m2 a = 1.0f;
  c10::Float8_e5m2 b = 2.0f;
  c10::Float8_e5m2 add = 3.0f;
  c10::Float8_e5m2 sub = -1.0f;
  c10::Float8_e5m2 mul = 2.0f;
  c10::Float8_e5m2 div = 0.5f;

  EXPECT_EQ(a + b, add);
  EXPECT_EQ(a - b, sub);
  EXPECT_EQ(a * b, mul);
  EXPECT_EQ(a / b, div);
}

TEST(TestDtype, TestFloat8_e5m2fnuz) {
  c10::Float8_e5m2fnuz a = 1.0f;
  c10::Float8_e5m2fnuz b = 2.0f;
  c10::Float8_e5m2fnuz add = 3.0f;
  c10::Float8_e5m2fnuz sub = -1.0f;
  c10::Float8_e5m2fnuz mul = 2.0f;
  c10::Float8_e5m2fnuz div = 0.5f;

  EXPECT_EQ(a + b, add);
  EXPECT_EQ(a - b, sub);
  EXPECT_EQ(a * b, mul);
  EXPECT_EQ(a / b, div);
}

TEST(TestDtype, TestHalf) {
  c10::Half a = 1.0f;
  c10::Half b = 2.0f;
  c10::Half add = 3.0f;
  c10::Half sub = -1.0f;
  c10::Half mul = 2.0f;
  c10::Half div = 0.5f;

  EXPECT_EQ(a + b, add);
  EXPECT_EQ(a - b, sub);
  EXPECT_EQ(a * b, mul);
  EXPECT_EQ(a / b, div);
}

TEST(TestDtype, TestComplexFloat) {
  c10::complex<float> a(std::complex<float>(1.0f, 2.0f));
  c10::complex<float> b(std::complex<float>(3.0f, 4.0f));
  c10::complex<float> add(std::complex<float>(4.0f, 6.0f));
  c10::complex<float> sub(std::complex<float>(-2.0f, -2.0f));
  c10::complex<float> mul(std::complex<float>(-5.0f, 10.0f));
  c10::complex<float> div(std::complex<float>(0.44f, 0.08f));

  EXPECT_EQ(a + b, add);
  EXPECT_EQ(a - b, sub);
  EXPECT_EQ(a * b, mul);
  EXPECT_EQ(a / b, div);
}

TEST(TestDtype, TestQuints) {
  auto a = torch::headeronly::quint8(5);
  auto b = torch::headeronly::quint4x2(1);
  auto c = torch::headeronly::quint2x4(5);

  EXPECT_EQ(a + a, torch::headeronly::quint8(10));
  EXPECT_EQ(b + b, torch::headeronly::quint4x2(2));
  EXPECT_EQ(c + c, torch::headeronly::quint2x4(10));
}

TEST(TestDtype, TestQintAndBits) {
  auto a = torch::headeronly::qint32(5);
  auto b = torch::headeronly::qint8(1);
  auto c = torch::headeronly::bits1x8(5);
  auto d = torch::headeronly::bits2x4(1);
  auto e = torch::headeronly::bits4x2(9);
  auto f = torch::headeronly::bits8(9);
  auto g = torch::headeronly::bits16(9);

  EXPECT_EQ(a + a, torch::headeronly::qint32(10));
  EXPECT_EQ(b + b, torch::headeronly::qint8(2));
  EXPECT_EQ(c + c, torch::headeronly::bits1x8(10));
  EXPECT_EQ(d * 2, torch::headeronly::bits2x4(2));
  EXPECT_EQ(e * 2, torch::headeronly::bits4x2(18));
  EXPECT_EQ(f - f, torch::headeronly::bits8(0));
  EXPECT_EQ(g + g, torch::headeronly::bits16(9));
}
