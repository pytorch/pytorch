#include <c10/util/BFloat16.h>
#include <gtest/gtest.h>

namespace {
  TEST(BFloat16Conversion, FloatToBFloat16AndBack) {
    float in[100];
    for (int i = 0; i < 100; ++i) {
      in[i] = i + 1.25;
    }

    c10::BFloat16 bfloats[100];
    float out[100];

    for (int i = 0; i < 100; ++i) {
      bfloats[i].val_ = c10::detail::bf16_from_f32(in[i]);
      out[i] = c10::detail::f32_from_bf16(bfloats[i].val_);

      // The relative error should be less than 1/(2^7) since bfloat16
      // has 7 bits mantissa.
      EXPECT_LE(fabs(out[i] - in[i]) / in[i], 1.0 / 128);
    }
  }
} // namespace
