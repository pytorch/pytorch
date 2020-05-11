#include <algorithm>
#include <cinttypes>
#include <cmath>

#include "fake_nnpi_ops_utils.h"

namespace caffe2 {

namespace fake_nnpi {

/// Decompose a float (<1) into (A / 2^N). Taken from ICE_REF code.
// https://github.com/IntelAI/nnpi-sw/blob/95b51ec28114a1ad16026d1df498fb29cca3cf8d/src/compiler/nnpi_compiler/src/ice_kernel_common/include/ice_kernel_common/ice_kernel_common_api.h#L76
class ScaleApproximation {
 public:
  ScaleApproximation() : m_a(0), m_n(0) {}
  explicit ScaleApproximation(
      float q,
      int scaleApproximationNumBits = kNUM_OF_BITS_FOR_Q_APPROX,
      bool limit = false) {
    m_a = m_n = 0;
    if (q == 0) {
      return;
    }
    float q_abs = fabs(q);
    for (int q_bits = scaleApproximationNumBits; q_bits > 0; q_bits--) {
      m_n = (int)floor(log2((float)((1 << q_bits) - 1) / q_abs));

      if (limit) {
        if (m_n > scaleApproximationNumBits)
          m_n = scaleApproximationNumBits;
      }
      // m_n will be almost always positive. But in case it is negative we need
      // to divide
      m_a = (m_n >= 0) ? (int)floor((float)q_abs * (1ll << m_n))
                       : (int)floor((float)q_abs / (1ll << (-m_n)));
      if (m_a <= kELTWISE_ULIMIT_SHORT) {
        break;
      }
    }
    if (q < 0)
      m_a = -m_a;
  }
  int m_a;
  int m_n;
  int getA() const {
    return m_a;
  }
  int getN() const {
    return m_n;
  }
};

/// \brief Fixed-point requantization. We approximate the float multiplier (<1)
/// by (A / 2^N) So what we want to compute is round(input_val * multiplier) +
/// outputOffset It's roughly ((input_val * A) >> N + outputOffset) with some
/// magic to take care of rounding, which I don't fully understand yet. The code
/// is taken from ICE-REF
/// (https://github.com/IntelAI/nnpi-sw/blob/8cc240a1bd77c6372fb6300c8bb157ae5954323e/src/compiler/nnpi_compiler/src/ice_layers/src/ice_layers_utils.cpp#L726).
/// I found https://fb.quip.com/7ta9AfOWMMuv and
/// https://github.com/pytorch/QNNPACK/blob/master/src/requantization/precise-scalar.c
/// useful to understand the basics of fixed-point requantization.
int8_t nnpiQuantize(
    int32_t input_val,
    float multiplier,
    int32_t outputOffset,
    bool round_bit_en,
    bool is_signed,
    bool round_half_to_nearest_up) {
  ScaleApproximation sa(multiplier, 8);
  uint8_t result_mult_int = sa.getA();
  uint32_t result_shift_val = sa.getN();
  int64_t temp = (int64_t)input_val * (int16_t)result_mult_int;

  if (result_shift_val != 0) {
    if (round_half_to_nearest_up) {
      temp >>= (result_shift_val - 1);
      temp += ((uint64_t)round_bit_en);
      temp &= 0x000001FFFFFFFFFEull;
      temp >>= 1;
    } else {
      // Following code is for HALF RND TO EVEN
      uint32_t mask = pow(2, result_shift_val - 1) - 1;
      bool sticky_bit = ((temp & mask) != 0);
      bool r = (temp >> (result_shift_val - 1)) & 0x01;
      bool l = (temp >> result_shift_val) & 0x01;
      temp >>= (result_shift_val);
      if (r & (l | sticky_bit))
        temp += ((uint64_t)round_bit_en);
      temp &= 0x000001FFFFFFFFFFull;
    }
  }

  temp += outputOffset;
  const bool sign_bit = (temp & 0x0000008000000000ull) != 0;

  if (is_signed) {
    if (sign_bit) {
      int8_t out_val = (int8_t)(1 << 7); // 0x80

      if ((temp & 0x0000007FFFFFFF80ull) == 0x0000007FFFFFFF80ull) {
        // no saturation needed - just "squeezing" sign extended bits
        out_val |= ((int8_t)(temp & 0x000000000000007Full));
      }
      // else - number remains 0x80

      return out_val;

    } else {
      const int8_t result_no_sat = (int8_t)(temp & 0x000000000000007Full);
      const bool saturation_set = ((temp & 0x0000007FFFFFFF80ull) != 0);

      int8_t out_val = saturation_set ? 0x7F : result_no_sat;
      return out_val;
    }
  } else {
    const bool saturation_set = ((temp & 0x000000FFFFFFFF00ull) != 0);
    const int8_t result_no_sat = (temp & 0x00000000000000FFull);

    // the "OLD" behavior - keeping it until cofirmed with Delphi folks
    //   const bool msb_bit = !(sign_bit);
    //   const uint8_t sign_pack_bits = sign_bit ? 0 : 0x7F;
    //   int8_t out_val = saturation_set ? (((int8_t)msb_bit << 7) |
    //   sign_pack_bits) : result_no_sat;

    int8_t out_val = saturation_set ? (sign_bit ? 0x00 : 0xFF) : result_no_sat;

    return out_val;
  }
}

/// @brief Reference implementation of matrix multiply with uint8 for A,
///  int8 for B^T with 32-bit accumulation, and outputs C in uint8.
void matmul_u8i8u8acc32_ref(
    int M,
    int N,
    int K,
    int lda,
    int ldb,
    int ldc,
    const uint8_t* A,
    int32_t A_zero_point,
    const int8_t* B,
    int32_t B_zero_point,
    const int32_t* bias,
    uint8_t* C,
    float C_multiplier, // A_scale * B_scale / C_scale
    int32_t C_zero_point,
    bool fuse_relu) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      int32_t sum = bias ? bias[j] : 0;
      for (int k = 0; k < K; ++k) {
        sum +=
            (A[i * lda + k] - A_zero_point) * (B[j * ldb + k] - B_zero_point);
      }
      /// Note that we are doing round-half-to-nearest-up here. Once we get next
      /// step hardware we probably want to change this.
      uint8_t rounded = static_cast<uint32_t>(
          nnpiQuantize(sum, C_multiplier, C_zero_point, true, false, true));
      C[i * ldc + j] =
          std::max(static_cast<uint8_t>(fuse_relu ? C_zero_point : 0), rounded);
    }
  }
}

} // namespace fake_nnpi
} // namespace caffe2
