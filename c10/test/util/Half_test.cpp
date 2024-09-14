#include <cmath>
#include <limits>
#include <vector>

#include <c10/util/Half.h>
#include <c10/util/floating_point_utils.h>
#include <c10/util/irange.h>
#include <gtest/gtest.h>

namespace {

float halfbits2float(unsigned short h) {
  unsigned sign = ((h >> 15) & 1);
  unsigned exponent = ((h >> 10) & 0x1f);
  unsigned mantissa = ((h & 0x3ff) << 13);

  if (exponent == 0x1f) { /* NaN or Inf */
    mantissa = (mantissa ? (sign = 0, 0x7fffff) : 0);
    exponent = 0xff;
  } else if (!exponent) { /* Denorm or Zero */
    if (mantissa) {
      // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
      unsigned int msb;
      exponent = 0x71;
      do {
        msb = (mantissa & 0x400000);
        mantissa <<= 1; /* normalize */
        --exponent;
      } while (!msb);
      mantissa &= 0x7fffff; /* 1.mantissa is implicit */
    }
  } else {
    exponent += 0x70;
  }

  unsigned result_bit = (sign << 31) | (exponent << 23) | mantissa;

  return c10::detail::fp32_from_bits(result_bit);
}

unsigned short float2halfbits(float src) {
  unsigned x = c10::detail::fp32_to_bits(src);

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables,cppcoreguidelines-avoid-magic-numbers)
  unsigned u = (x & 0x7fffffff), remainder, shift, lsb, lsb_s1, lsb_m1;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  unsigned sign, exponent, mantissa;

  // Get rid of +NaN/-NaN case first.
  if (u > 0x7f800000) {
    return 0x7fffU;
  }

  sign = ((x >> 16) & 0x8000);

  // Get rid of +Inf/-Inf, +0/-0.
  if (u > 0x477fefff) {
    return sign | 0x7c00U;
  }
  if (u < 0x33000001) {
    return (sign | 0x0000);
  }

  exponent = ((u >> 23) & 0xff);
  mantissa = (u & 0x7fffff);

  if (exponent > 0x70) {
    shift = 13;
    exponent -= 0x70;
  } else {
    shift = 0x7e - exponent;
    exponent = 0;
    mantissa |= 0x800000;
  }
  lsb = (1 << shift);
  lsb_s1 = (lsb >> 1);
  lsb_m1 = (lsb - 1);

  // Round to nearest even.
  remainder = (mantissa & lsb_m1);
  mantissa >>= shift;
  if (remainder > lsb_s1 || (remainder == lsb_s1 && (mantissa & 0x1))) {
    ++mantissa;
    if (!(mantissa & 0x3ff)) {
      ++exponent;
      mantissa = 0;
    }
  }

  return (sign | (exponent << 10) | mantissa);
}
TEST(HalfConversionTest, TestPorableConversion) {
  std::vector<uint16_t> inputs = {
      0,
      0xfbff, // 1111 1011 1111 1111
      (1 << 15 | 1),
      0x7bff // 0111 1011 1111 1111
  };
  for (auto x : inputs) {
    auto target = c10::detail::fp16_ieee_to_fp32_value(x);
    EXPECT_EQ(halfbits2float(x), target)
        << "Test failed for uint16 to float " << x << "\n";
    EXPECT_EQ(
        float2halfbits(target), c10::detail::fp16_ieee_from_fp32_value(target))
        << "Test failed for float to uint16" << target << "\n";
  }
}

TEST(HalfConversion, TestNativeConversionToFloat) {
  // There are only 2**16 possible values, so test them all
  for (auto x : c10::irange(std::numeric_limits<uint16_t>::max() + 1)) {
    auto h = c10::Half(x, c10::Half::from_bits());
    auto f = halfbits2float(x);
    // NaNs are not equal to each other
    if (std::isnan(f) && std::isnan(static_cast<float>(h))) {
      continue;
    }
    EXPECT_EQ(f, static_cast<float>(h)) << "Conversion error using " << x;
  }
}

TEST(HalfConversion, TestNativeConversionToHalf) {
  auto check_conversion = [](float f) {
    auto h = c10::Half(f);
    auto h_bits = float2halfbits(f);
    // NaNs are not equal to each other, just check that half is NaN
    if (std::isnan(f)) {
      EXPECT_TRUE(std::isnan(static_cast<float>(h)));
    } else {
      EXPECT_EQ(h.x, h_bits) << "Conversion error using " << f;
    }
  };

  for (auto x : c10::irange(std::numeric_limits<uint16_t>::max() + 1)) {
    check_conversion(halfbits2float(x));
  }
  // Check a few values outside of Half range
  check_conversion(std::numeric_limits<float>::max());
  check_conversion(std::numeric_limits<float>::min());
  check_conversion(std::numeric_limits<float>::epsilon());
  check_conversion(std::numeric_limits<float>::lowest());
}

} // namespace
