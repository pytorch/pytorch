#include <c10/util/BFloat16.h>
#include <gtest/gtest.h>
/*
namespace {
  float float_from_bytes(
      uint32_t sign,
      uint32_t exponent,
      uint32_t fraction
  ) {
      uint32_t bytes;
      bytes = 0;
      bytes |= sign;
      bytes <<= 8;
      bytes |= exponent;
      bytes <<= 23;
      bytes |= fraction;

      float res;
      std::memcpy(&res, &bytes, sizeof(res));
      return res;
  }

  TEST(BFloat16Conversion, FloatToBFloat16AndBack) {
    float in[100];
    for (int i = 0; i < 100; ++i) {
      in[i] = i + 1.25;
    }

    c10::BFloat16 bfloats[100];
    float out[100];

    for (int i = 0; i < 100; ++i) {
      bfloats[i].val_ = c10::detail::bits_from_f32(in[i]);
      out[i] = c10::detail::f32_from_bits(bfloats[i].val_);

      // The relative error should be less than 1/(2^7) since bfloat16
      // has 7 bits mantissa.
      EXPECT_LE(fabs(out[i] - in[i]) / in[i], 1.0 / 128);
    }
  }

  TEST(BFloat16Conversion, NaN) {
    float inNaN = float_from_bytes(0, 0xFF, 0x7FFFFF);
    EXPECT_TRUE(std::isnan(inNaN));

    c10::BFloat16 a = c10::BFloat16(inNaN);
    float out = c10::detail::f32_from_bits(a.val_);

    EXPECT_TRUE(std::isnan(out));
  }

  TEST(BFloat16Conversion, Inf) {
    float inInf = float_from_bytes(0, 0xFF, 0);
    EXPECT_TRUE(std::isinf(inInf));

    c10::BFloat16 a = c10::BFloat16(inInf);
    float out = c10::detail::f32_from_bits(a.val_);

    EXPECT_TRUE(std::isinf(out));
  }

  TEST(BFloat16Conversion, SmallestDenormal) {
    float in =  std::numeric_limits<float>::denorm_min(); // The smallest non-zero subnormal number
    c10::BFloat16 a = c10::BFloat16(in);
    float out = c10::detail::f32_from_bits(a.val_);

    EXPECT_FLOAT_EQ(in, out);
  }

  TEST(BFloat16Conversion, BiggestDenormal) {
    float in =  float_from_bytes(0, 0, 0x7FFFFF); // The largest subnormal number
    c10::BFloat16 a = c10::BFloat16(in);
    float out = c10::detail::f32_from_bits(a.val_);

    EXPECT_FLOAT_EQ(1.1663108e-38, out);
  }
} // namespace
*/
