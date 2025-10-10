#include <gtest/gtest.h>
#include <torch/headeronly/cpu/vec/vec_half.h>
#include <torch/headeronly/util/Half.h>

TEST(TestVecHalf, TestConversion) {
  float f32s[100];
  for (int i = 0; i < 100; i++) {
    f32s[i] = static_cast<float>(i + 0.3);
  }
  for (int i = 0; i < 100; i++) {
#if (defined(CPU_CAPABILITY_AVX2) || defined(CPU_CAPABILITY_AVX512)) && \
    !defined(__APPLE__)
    uint16_t u16 = torch::headeronly::vec::float2half_scalar(f32s[i]);
    float x = torch::headeronly::vec::half2float_scalar(u16);
    EXPECT_EQ(
        u16, torch::headeronly::detail::fp16_ieee_from_fp32_value(f32s[i]))
        << "Test failed for float to uint16 " << f32s[i] << "\n";
    EXPECT_EQ(x, torch::headeronly::detail::fp16_ieee_to_fp32_value(u16))
        << "Test failed for uint16 to float " << u16 << "\n";
#endif
  }
}
