/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <assert.h>
#include <stdint.h>

#include <fp16/bitcasts.h>
#include <qnnpack/requantization-stubs.h>
#include <qnnpack/scalar-utils.h>

void pytorch_qnnp_requantize_precise__scalar_unsigned32(
    size_t n,
    const int32_t* input,
    float scale,
    uint8_t zero_point,
    uint8_t qmin,
    uint8_t qmax,
    uint8_t* output) {
  assert(n % 4 == 0);
  assert(scale < 1.0f);
  assert(scale >= 0x1.0p-32f);

  const uint32_t scale_bits = fp32_to_bits(scale);
  const uint32_t multiplier = (scale_bits << 8) | UINT32_C(0x80000000);
  const uint32_t shift = 127 + 31 - (scale_bits >> 23);
  assert(shift >= 32);
  assert(shift < 64);

  const uint64_t rounding = UINT64_C(1) << (shift - 1);
  const uint32_t rounding_hi = (uint32_t)(rounding >> 32);
  const uint32_t rounding_lo = (uint32_t)rounding;
  const uint32_t shift_minus_32 = shift - 32;
  const int32_t smin = (int32_t)(uint32_t)qmin - (int32_t)(uint32_t)zero_point;
  const int32_t smax = (int32_t)(uint32_t)qmax - (int32_t)(uint32_t)zero_point;
  for (; n != 0; n -= 4) {
    const int32_t x = input[0];
    const int32_t y = input[1];
    const int32_t z = input[2];
    const int32_t w = input[3];
    input += 4;

    /*
     * Compute absolute value of input as unsigned 32-bit int.
     * All further computations will work with unsigned values to avoid
     * undefined behaviour on signed operations.
     */
    const uint32_t x_abs = (x >= 0) ? (uint32_t)x : -(uint32_t)x;
    const uint32_t y_abs = (y >= 0) ? (uint32_t)y : -(uint32_t)y;
    const uint32_t z_abs = (z >= 0) ? (uint32_t)z : -(uint32_t)z;
    const uint32_t w_abs = (w >= 0) ? (uint32_t)w : -(uint32_t)w;

    /* Compute full 64-bit product of 32-bit factors */
    const uint64_t x_product = (uint64_t)x_abs * (uint64_t)multiplier;
    const uint64_t y_product = (uint64_t)y_abs * (uint64_t)multiplier;
    const uint64_t z_product = (uint64_t)z_abs * (uint64_t)multiplier;
    const uint64_t w_product = (uint64_t)w_abs * (uint64_t)multiplier;

    /*
     * Shift the full 64-bit product right with rounding.
     * Rounding is performed towards closest integer, with midpoints rounded up
     * (same as away from zero).
     *
     * Generally, this operation requires both 64-bit addition and 64-bit shift,
     * but we use two tricks to replace 64-bit operations with 32-bit
     * operations.
     *
     * To avoid full 64-bit addition we make use of three facts:
     * - 64-bit rounding value added before the shift is a power of 2, and thus
     * has only one bit set.
     * - When 0x1.0p-32f <= scale < 0x1.0p-31f, then the non-zero bit in
     * rounding is in the low 32 bits, and rounding is exactly 0x80000000
     * (2**31), because rounding is 2**(scale-1) and scale >= 32. In this case,
     *   addition of rounding can affect high 32 bits of the product only
     * through overflow, which happens if low 32-bit part of the product equals
     * or exceeds 0x80000000. We can reformulate the latter condition as low
     * 32-bit part of the product has the bit 31 set, and then overflow happens
     * if both the low 32-bit part of the product and the low 32-bit part of the
     * rounding value have bit 31 set. Since 32-bit numbers with the bit 31 set
     * are negative when interpreted as signed integers, we can check the
     * overflow condition as (int32_t) (LOW(product) & LOW(rounding)) < 0
     * - When 0x1.0p-31f <= scale < 1.0f, then the non-zero bit is in the high
     * 32 bits of rounding. We just need to do 32-bit addition of high 32 bits
     * of rounding and high 32 bits of product. This addition never overflows
     * because product <= 0x80000000 * 0xFFFFFF00 < 2**63 and rounding =
     * 2**(scale-1) <= 2**62.
     *
     * To avoid full 64-bit shift, we leverage the fact that shift >= 32, and do
     * it in two steps:
     * - Shift by 32, which can be implemented by extracting the high 32-bit word
     * on 32-bit systems.
     * - Shift by (shift - 32), which can be implemented as a 32-bit shift of
     * high word of addition result.
     */
    const uint32_t x_carry_lo =
        (uint32_t)((int32_t)((uint32_t)x_product & rounding_lo) < 0);
    const uint32_t y_carry_lo =
        (uint32_t)((int32_t)((uint32_t)y_product & rounding_lo) < 0);
    const uint32_t z_carry_lo =
        (uint32_t)((int32_t)((uint32_t)z_product & rounding_lo) < 0);
    const uint32_t w_carry_lo =
        (uint32_t)((int32_t)((uint32_t)w_product & rounding_lo) < 0);

    const uint32_t x_product_hi = (uint32_t)(x_product >> 32);
    const uint32_t y_product_hi = (uint32_t)(y_product >> 32);
    const uint32_t z_product_hi = (uint32_t)(z_product >> 32);
    const uint32_t w_product_hi = (uint32_t)(w_product >> 32);

    const uint32_t x_abs_scaled =
        (uint32_t)(x_product_hi + rounding_hi + x_carry_lo) >> shift_minus_32;
    const uint32_t y_abs_scaled =
        (uint32_t)(y_product_hi + rounding_hi + y_carry_lo) >> shift_minus_32;
    const uint32_t z_abs_scaled =
        (uint32_t)(z_product_hi + rounding_hi + z_carry_lo) >> shift_minus_32;
    const uint32_t w_abs_scaled =
        (uint32_t)(w_product_hi + rounding_hi + w_carry_lo) >> shift_minus_32;

    /* Copy the sign of input to scaled absolute input value */
    const int32_t x_scaled = (int32_t)(x >= 0 ? x_abs_scaled : -x_abs_scaled);
    const int32_t y_scaled = (int32_t)(y >= 0 ? y_abs_scaled : -y_abs_scaled);
    const int32_t z_scaled = (int32_t)(z >= 0 ? z_abs_scaled : -z_abs_scaled);
    const int32_t w_scaled = (int32_t)(w >= 0 ? w_abs_scaled : -w_abs_scaled);

    /*
     * Clamp scaled value with zero point between (qmin - zero point) and (qmax
     * - zero point).
     */
    const int32_t x_clamped =
        x_scaled < smin ? smin : x_scaled > smax ? smax : x_scaled;
    const int32_t y_clamped =
        y_scaled < smin ? smin : y_scaled > smax ? smax : y_scaled;
    const int32_t z_clamped =
        z_scaled < smin ? smin : z_scaled > smax ? smax : z_scaled;
    const int32_t w_clamped =
        w_scaled < smin ? smin : w_scaled > smax ? smax : w_scaled;

    /*
     * Add zero point to clamped value.
     * The result is guaranteed to be in [qmin, qmax] range.
     *
     * This addition can not be safely done before clamping, because scaled
     * values are in [-2147483520, 2147483519] range, so addition of zero point
     * (which can be up to 255) can overflow signed 32-bit integer.
     */
    const int32_t x_biased = x_clamped + zero_point;
    const int32_t y_biased = y_clamped + zero_point;
    const int32_t z_biased = z_clamped + zero_point;
    const int32_t w_biased = w_clamped + zero_point;

    output[0] = (uint8_t)x_biased;
    output[1] = (uint8_t)y_biased;
    output[2] = (uint8_t)z_biased;
    output[3] = (uint8_t)w_biased;
    output += 4;
  }
}

void pytorch_qnnp_requantize_precise__scalar_unsigned64(
    size_t n,
    const int32_t* input,
    float scale,
    uint8_t zero_point,
    uint8_t qmin,
    uint8_t qmax,
    uint8_t* output) {
  assert(n % 4 == 0);
  assert(scale < 1.0f);
  assert(scale >= 0x1.0p-32f);

  const uint32_t scale_bits = fp32_to_bits(scale);
  const uint32_t multiplier =
      (scale_bits & UINT32_C(0x007FFFFF)) | UINT32_C(0x00800000);
  const uint32_t shift = 127 + 23 - (scale_bits >> 23);
  assert(shift >= 24);
  assert(shift < 56);

  const uint64_t rounding = UINT64_C(1) << (shift - 1);
  const int32_t smin = (int32_t)(uint32_t)qmin - (int32_t)(uint32_t)zero_point;
  const int32_t smax = (int32_t)(uint32_t)qmax - (int32_t)(uint32_t)zero_point;
  for (; n != 0; n -= 4) {
    const int32_t x = input[0];
    const int32_t y = input[1];
    const int32_t z = input[2];
    const int32_t w = input[3];
    input += 4;

    /*
     * Compute absolute value of input as unsigned 32-bit int.
     * All further computations will work with unsigned values to avoid
     * undefined behaviour on signed operations.
     */
    const uint32_t x_abs = (x >= 0) ? (uint32_t)x : -(uint32_t)x;
    const uint32_t y_abs = (y >= 0) ? (uint32_t)y : -(uint32_t)y;
    const uint32_t z_abs = (z >= 0) ? (uint32_t)z : -(uint32_t)z;
    const uint32_t w_abs = (w >= 0) ? (uint32_t)w : -(uint32_t)w;

    /* Compute full 64-bit product of 32-bit factors */
    const uint64_t x_product = (uint64_t)x_abs * (uint64_t)multiplier;
    const uint64_t y_product = (uint64_t)y_abs * (uint64_t)multiplier;
    const uint64_t z_product = (uint64_t)z_abs * (uint64_t)multiplier;
    const uint64_t w_product = (uint64_t)w_abs * (uint64_t)multiplier;

    /*
     * Shift the full 64-bit product right with rounding.
     * Rounding is performed towards closest integer, with midpoints rounded up
     * (same as away from zero).
     *
     * Note that although rounding is precomputed, it is dependent on shift
     * value, and on processors with 64-bit "right shift with rounding"
     * instruction each line below can be represented by just one such
     * instruction (e.g. VRSHL.U64 on ARM NEON, URSHL in ARM64 Advanced SIMD).
     */
    const uint32_t x_abs_scaled = (uint32_t)((x_product + rounding) >> shift);
    const uint32_t y_abs_scaled = (uint32_t)((y_product + rounding) >> shift);
    const uint32_t z_abs_scaled = (uint32_t)((z_product + rounding) >> shift);
    const uint32_t w_abs_scaled = (uint32_t)((w_product + rounding) >> shift);

    /*
     * Copy the sign of input to scaled absolute input value.
     *
     * On x86 processors with SSSE3 instruction set, this operation nicely maps
     * to PSIGND instruction.
     */
    const int32_t x_scaled = (int32_t)(x >= 0 ? x_abs_scaled : -x_abs_scaled);
    const int32_t y_scaled = (int32_t)(y >= 0 ? y_abs_scaled : -y_abs_scaled);
    const int32_t z_scaled = (int32_t)(z >= 0 ? z_abs_scaled : -z_abs_scaled);
    const int32_t w_scaled = (int32_t)(w >= 0 ? w_abs_scaled : -w_abs_scaled);

    /*
     * Clamp scaled value with zero point between (qmin - zero point) and (qmax
     * - zero point).
     */
    const int32_t x_clamped =
        x_scaled < smin ? smin : x_scaled > smax ? smax : x_scaled;
    const int32_t y_clamped =
        y_scaled < smin ? smin : y_scaled > smax ? smax : y_scaled;
    const int32_t z_clamped =
        z_scaled < smin ? smin : z_scaled > smax ? smax : z_scaled;
    const int32_t w_clamped =
        w_scaled < smin ? smin : w_scaled > smax ? smax : w_scaled;

    /*
     * Add zero point to clamped value.
     * The result is guaranteed to be in [qmin, qmax] range.
     *
     * This addition can not be safely done before clamping, because scaled
     * values are in [-2147483520, 2147483519] range, so addition of zero point
     * (which can be up to 255) can overflow signed 32-bit integer.
     */
    const int32_t x_biased = x_clamped + zero_point;
    const int32_t y_biased = y_clamped + zero_point;
    const int32_t z_biased = z_clamped + zero_point;
    const int32_t w_biased = w_clamped + zero_point;

    output[0] = (uint8_t)x_biased;
    output[1] = (uint8_t)y_biased;
    output[2] = (uint8_t)z_biased;
    output[3] = (uint8_t)w_biased;
    output += 4;
  }
}

void pytorch_qnnp_requantize_precise__scalar_signed64(
    size_t n,
    const int32_t* input,
    float scale,
    uint8_t zero_point,
    uint8_t qmin,
    uint8_t qmax,
    uint8_t* output) {
  assert(n % 4 == 0);
  assert(scale < 1.0f);
  assert(scale >= 0x1.0p-32f);

  const uint32_t scale_bits = fp32_to_bits(scale);
  const int32_t multiplier =
      ((int32_t)scale_bits & INT32_C(0x007FFFFF)) | INT32_C(0x00800000);
  const uint32_t shift = 127 + 23 - (scale_bits >> 23);
  assert(shift >= 24);
  assert(shift < 56);

  const int64_t rounding = INT64_C(1) << (shift - 1);
  const int32_t smin = (int32_t)(uint32_t)qmin - (int32_t)(uint32_t)zero_point;
  const int32_t smax = (int32_t)(uint32_t)qmax - (int32_t)(uint32_t)zero_point;
  for (; n != 0; n -= 4) {
    const int32_t x = input[0];
    const int32_t y = input[1];
    const int32_t z = input[2];
    const int32_t w = input[3];
    input += 4;

    /*
     * Compute full 64-bit product of signed 32-bit factors.
     *
     * Note: multiplier can be treated as either signed or unsigned.
     */
    const int64_t x_product = (int64_t)x * (int64_t)multiplier;
    const int64_t y_product = (int64_t)y * (int64_t)multiplier;
    const int64_t z_product = (int64_t)z * (int64_t)multiplier;
    const int64_t w_product = (int64_t)w * (int64_t)multiplier;

    /*
     * Adjust product before subsequent shift with rounding up to simulate shift
     * with rounding away from zero.
     */
    const int64_t x_adjusted_product = x_product - (int64_t)(x < 0);
    const int64_t y_adjusted_product = y_product - (int64_t)(y < 0);
    const int64_t z_adjusted_product = z_product - (int64_t)(z < 0);
    const int64_t w_adjusted_product = w_product - (int64_t)(w < 0);

    /*
     * Arithmetically shift the full 64-bit product right with rounding.
     * Rounding is performed towards closest integer, with midpoints rounded up.
     *
     * Note that although rounding is precomputed, it is dependent on shift
     * value, and on processors with 64-bit "right shift with rounding"
     * instruction each line below can be represented by just one such
     * instruction (e.g. VRSHL.S64 on ARM NEON, SRSHL in ARM64 Advanced SIMD).
     */
    const int32_t x_scaled =
        (int32_t)asr_s64(x_adjusted_product + rounding, shift);
    const int32_t y_scaled =
        (int32_t)asr_s64(y_adjusted_product + rounding, shift);
    const int32_t z_scaled =
        (int32_t)asr_s64(z_adjusted_product + rounding, shift);
    const int32_t w_scaled =
        (int32_t)asr_s64(w_adjusted_product + rounding, shift);

    /*
     * Clamp scaled value with zero point between (qmin - zero point) and (qmax
     * - zero point).
     */
    const int32_t x_clamped =
        x_scaled < smin ? smin : x_scaled > smax ? smax : x_scaled;
    const int32_t y_clamped =
        y_scaled < smin ? smin : y_scaled > smax ? smax : y_scaled;
    const int32_t z_clamped =
        z_scaled < smin ? smin : z_scaled > smax ? smax : z_scaled;
    const int32_t w_clamped =
        w_scaled < smin ? smin : w_scaled > smax ? smax : w_scaled;

    /*
     * Add zero point to clamped value.
     * The result is guaranteed to be in [qmin, qmax] range.
     *
     * This addition can not be safely done before clamping, because scaled
     * values are in [-2147483520, 2147483519] range, so addition of zero point
     * (which can be up to 255) can overflow signed 32-bit integer.
     */
    const int32_t x_biased = x_clamped + zero_point;
    const int32_t y_biased = y_clamped + zero_point;
    const int32_t z_biased = z_clamped + zero_point;
    const int32_t w_biased = w_clamped + zero_point;

    output[0] = (uint8_t)x_biased;
    output[1] = (uint8_t)y_biased;
    output[2] = (uint8_t)z_biased;
    output[3] = (uint8_t)w_biased;
    output += 4;
  }
}
