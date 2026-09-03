// clang-format off
#include <c10/util/BFloat16.h>
#include <c10/util/BFloat16-math.h>
#include <c10/util/irange.h>
// clang-format on
#include <gtest/gtest.h>

namespace {
float float_from_bytes(uint32_t sign, uint32_t exponent, uint32_t fraction) {
  uint32_t bytes = 0;
  bytes |= sign;
  bytes <<= 8;
  bytes |= exponent;
  bytes <<= 23;
  bytes |= fraction;

  float res = 0;
  std::memcpy(&res, &bytes, sizeof(res));
  return res;
}

TEST(BFloat16Conversion, FloatToBFloat16AndBack) {
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,cppcoreguidelines-avoid-magic-numbers,modernize-avoid-c-arrays)
  float in[100];
  for (const auto i : c10::irange(100)) {
    // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,cppcoreguidelines-avoid-magic-numbers)
    in[i] = i + 1.25;
  }

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,cppcoreguidelines-avoid-magic-numbers,modernize-avoid-c-arrays)
  c10::BFloat16 bfloats[100];
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,cppcoreguidelines-avoid-magic-numbers,modernize-avoid-c-arrays)
  float out[100];

  for (const auto i : c10::irange(100)) {
    bfloats[i].x = c10::detail::bits_from_f32(in[i]);
    out[i] = c10::detail::f32_from_bits(bfloats[i].x);

    // The relative error should be less than 1/(2^7) since BFloat16
    // has 7 bits mantissa.
    EXPECT_LE(std::fabs(out[i] - in[i]) / in[i], 1.0 / 128);
  }
}

TEST(BFloat16Conversion, FloatToBFloat16RNEAndBack) {
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,cppcoreguidelines-avoid-magic-numbers,modernize-avoid-c-arrays)
  float in[100];
  for (const auto i : c10::irange(100)) {
    // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,cppcoreguidelines-avoid-magic-numbers)
    in[i] = i + 1.25;
  }

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,cppcoreguidelines-avoid-magic-numbers,modernize-avoid-c-arrays)
  c10::BFloat16 bfloats[100];
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,cppcoreguidelines-avoid-magic-numbers,modernize-avoid-c-arrays)
  float out[100];

  for (const auto i : c10::irange(100)) {
    bfloats[i].x = c10::detail::round_to_nearest_even(in[i]);
    out[i] = c10::detail::f32_from_bits(bfloats[i].x);

    // The relative error should be less than 1/(2^7) since BFloat16
    // has 7 bits mantissa.
    EXPECT_LE(std::fabs(out[i] - in[i]) / in[i], 1.0 / 128);
  }
}

TEST(BFloat16Conversion, NaN) {
  float inNaN = float_from_bytes(0, 0xFF, 0x7FFFFF);
  EXPECT_TRUE(std::isnan(inNaN));

  c10::BFloat16 a = c10::BFloat16(inNaN);
  float out = c10::detail::f32_from_bits(a.x);

  EXPECT_TRUE(std::isnan(out));
}

TEST(BFloat16Conversion, Inf) {
  float inInf = float_from_bytes(0, 0xFF, 0);
  EXPECT_TRUE(std::isinf(inInf));

  c10::BFloat16 a = c10::BFloat16(inInf);
  float out = c10::detail::f32_from_bits(a.x);

  EXPECT_TRUE(std::isinf(out));
}

TEST(BFloat16Conversion, SmallestDenormal) {
  float in = std::numeric_limits<float>::denorm_min(); // The smallest non-zero
                                                       // subnormal number
  c10::BFloat16 a = c10::BFloat16(in);
  float out = c10::detail::f32_from_bits(a.x);

  EXPECT_FLOAT_EQ(in, out);
}

TEST(BFloat16Math, Addition) {
  // This test verifies that if only first 7 bits of float's mantissa are
  // changed after addition, we should have no loss in precision.

  // input bits
  // S | Exponent | Mantissa
  // 0 | 10000000 | 10010000000000000000000 = 3.125
  float input = float_from_bytes(0, 0, 0x40480000);

  // expected bits
  // S | Exponent | Mantissa
  // 0 | 10000001 | 10010000000000000000000 = 6.25
  float expected = float_from_bytes(0, 0, 0x40c80000);

  c10::BFloat16 b{};
  b.x = c10::detail::bits_from_f32(input);
  b = b + b;

  float res = c10::detail::f32_from_bits(b.x);
  EXPECT_EQ(res, expected);
}

TEST(BFloat16Math, Subtraction) {
  // This test verifies that if only first 7 bits of float's mantissa are
  // changed after subtraction, we should have no loss in precision.

  // input bits
  // S | Exponent | Mantissa
  // 0 | 10000001 | 11101000000000000000000 = 7.625
  float input = float_from_bytes(0, 0, 0x40f40000);

  // expected bits
  // S | Exponent | Mantissa
  // 0 | 10000000 | 01010000000000000000000 = 2.625
  float expected = float_from_bytes(0, 0, 0x40280000);

  c10::BFloat16 b{};
  b.x = c10::detail::bits_from_f32(input);
  b = b - 5;

  float res = c10::detail::f32_from_bits(b.x);
  EXPECT_EQ(res, expected);
}

TEST(BFloat16Math, NextAfterZero) {
  const c10::BFloat16 zero{0};

  auto check_nextafter =
      [](c10::BFloat16 from, c10::BFloat16 to, c10::BFloat16 expected) {
        c10::BFloat16 actual = std::nextafter(from, to);
        // Check for bitwise equality!
        ASSERT_EQ(actual.x ^ expected.x, uint16_t{0});
      };
  check_nextafter(zero, zero, /*expected=*/zero);
  check_nextafter(zero, -zero, /*expected=*/-zero);
  check_nextafter(-zero, zero, /*expected=*/zero);
  check_nextafter(-zero, -zero, /*expected=*/-zero);
}

float BinaryToFloat(uint32_t bytes) {
  float res = 0;
  std::memcpy(&res, &bytes, sizeof(res));
  return res;
}

struct BFloat16TestParam {
  uint32_t input;
  uint16_t rne;
};

class BFloat16Test : public ::testing::Test,
                     public ::testing::WithParamInterface<BFloat16TestParam> {};

TEST_P(BFloat16Test, BFloat16RNETest) {
  float value = BinaryToFloat(GetParam().input);
  uint16_t rounded = c10::detail::round_to_nearest_even(value);
  EXPECT_EQ(GetParam().rne, rounded);
}

INSTANTIATE_TEST_SUITE_P(
    BFloat16TestInstantiation,
    BFloat16Test,
    ::testing::Values(
        BFloat16TestParam{0x3F848000, 0x3F84},
        BFloat16TestParam{0x3F848010, 0x3F85},
        BFloat16TestParam{0x3F850000, 0x3F85},
        BFloat16TestParam{0x3F858000, 0x3F86},
        BFloat16TestParam{0x3FFF8000, 0x4000}));

} // namespace
