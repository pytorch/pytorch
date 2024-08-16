/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <assert.h>
#include <stdint.h>

#include <psimd.h>

#include <fp16/bitcasts.h>
#include <qnnpack/requantization-stubs.h>

void pytorch_qnnp_requantize_precise__psimd(
    size_t n,
    const int32_t* input,
    float scale,
    uint8_t zero_point,
    uint8_t qmin,
    uint8_t qmax,
    uint8_t* output) {
  assert(n % 16 == 0);
  assert(scale < 1.0f);
  assert(scale >= 0x1.0p-32f);

  const uint32_t scale_bits = fp32_to_bits(scale);
  const uint32_t multiplier = (scale_bits << 8) | UINT32_C(0x80000000);
  const uint32_t shift = 127 + 31 - (scale_bits >> 23);
  assert(shift >= 32);
  assert(shift < 64);
  const uint64_t rounding = UINT64_C(1) << (shift - 1);

  const psimd_u32 vmultiplier_lo =
      psimd_splat_u32(multiplier & UINT32_C(0x0000FFFF));
  const psimd_u32 vmultiplier_hi = psimd_splat_u32(multiplier >> 16);
  const psimd_s32 vzero_point = psimd_splat_s32((int32_t)(uint32_t)zero_point);
  const psimd_s32 vsmin =
      psimd_splat_s32((int32_t)(uint32_t)qmin - (int32_t)(uint32_t)zero_point);
  const psimd_s32 vsmax =
      psimd_splat_s32((int32_t)(uint32_t)qmax - (int32_t)(uint32_t)zero_point);
  const psimd_u32 vrounding_lo = psimd_splat_u32((uint32_t)rounding);
  const psimd_u32 vrounding_hi = psimd_splat_u32((uint32_t)(rounding >> 32));
  const psimd_u32 vshift = psimd_splat_u32(shift - 32);
  for (; n != 0; n -= 16) {
    const psimd_s32 x = psimd_load_s32(input);
    const psimd_s32 y = psimd_load_s32(input + 4);
    const psimd_s32 z = psimd_load_s32(input + 8);
    const psimd_s32 w = psimd_load_s32(input + 12);
    input += 16;

    const psimd_s32 x_neg_mask = x >> psimd_splat_s32(31);
    const psimd_s32 y_neg_mask = y >> psimd_splat_s32(31);
    const psimd_s32 z_neg_mask = z >> psimd_splat_s32(31);
    const psimd_s32 w_neg_mask = w >> psimd_splat_s32(31);

    const psimd_u32 x_abs = (psimd_u32)((x ^ x_neg_mask) - x_neg_mask);
    const psimd_u32 y_abs = (psimd_u32)((y ^ y_neg_mask) - y_neg_mask);
    const psimd_u32 z_abs = (psimd_u32)((z ^ z_neg_mask) - z_neg_mask);
    const psimd_u32 w_abs = (psimd_u32)((w ^ w_neg_mask) - w_neg_mask);

    const psimd_u32 x_abs_lo = x_abs & psimd_splat_u32(UINT32_C(0x0000FFFF));
    const psimd_u32 x_abs_hi = x_abs >> psimd_splat_u32(16);
    const psimd_u32 y_abs_lo = y_abs & psimd_splat_u32(UINT32_C(0x0000FFFF));
    const psimd_u32 y_abs_hi = y_abs >> psimd_splat_u32(16);
    const psimd_u32 z_abs_lo = z_abs & psimd_splat_u32(UINT32_C(0x0000FFFF));
    const psimd_u32 z_abs_hi = z_abs >> psimd_splat_u32(16);
    const psimd_u32 w_abs_lo = w_abs & psimd_splat_u32(UINT32_C(0x0000FFFF));
    const psimd_u32 w_abs_hi = w_abs >> psimd_splat_u32(16);

    const psimd_u32 x_product_ll = x_abs_lo * vmultiplier_lo;
    const psimd_u32 y_product_ll = y_abs_lo * vmultiplier_lo;
    const psimd_u32 z_product_ll = z_abs_lo * vmultiplier_lo;
    const psimd_u32 w_product_ll = w_abs_lo * vmultiplier_lo;

    const psimd_u32 x_product_lh =
        x_abs_lo * vmultiplier_hi + (x_product_ll >> psimd_splat_u32(16));
    const psimd_u32 y_product_lh =
        y_abs_lo * vmultiplier_hi + (y_product_ll >> psimd_splat_u32(16));
    const psimd_u32 z_product_lh =
        z_abs_lo * vmultiplier_hi + (z_product_ll >> psimd_splat_u32(16));
    const psimd_u32 w_product_lh =
        w_abs_lo * vmultiplier_hi + (w_product_ll >> psimd_splat_u32(16));

    const psimd_u32 x_product_hl = x_abs_hi * vmultiplier_lo +
        (x_product_lh & psimd_splat_u32(UINT32_C(0x0000FFFF)));
    const psimd_u32 y_product_hl = y_abs_hi * vmultiplier_lo +
        (y_product_lh & psimd_splat_u32(UINT32_C(0x0000FFFF)));
    const psimd_u32 z_product_hl = z_abs_hi * vmultiplier_lo +
        (z_product_lh & psimd_splat_u32(UINT32_C(0x0000FFFF)));
    const psimd_u32 w_product_hl = w_abs_hi * vmultiplier_lo +
        (w_product_lh & psimd_splat_u32(UINT32_C(0x0000FFFF)));

    const psimd_u32 x_product_lo = (x_product_hl << psimd_splat_u32(16)) +
        (x_product_ll & psimd_splat_u32(UINT32_C(0x0000FFFF)));
    const psimd_u32 y_product_lo = (y_product_hl << psimd_splat_u32(16)) +
        (y_product_ll & psimd_splat_u32(UINT32_C(0x0000FFFF)));
    const psimd_u32 z_product_lo = (z_product_hl << psimd_splat_u32(16)) +
        (z_product_ll & psimd_splat_u32(UINT32_C(0x0000FFFF)));
    const psimd_u32 w_product_lo = (w_product_hl << psimd_splat_u32(16)) +
        (w_product_ll & psimd_splat_u32(UINT32_C(0x0000FFFF)));

    const psimd_u32 x_product_hi = x_abs_hi * vmultiplier_hi +
        (x_product_lh >> psimd_splat_u32(16)) +
        (x_product_hl >> psimd_splat_u32(16));
    const psimd_u32 y_product_hi = y_abs_hi * vmultiplier_hi +
        (y_product_lh >> psimd_splat_u32(16)) +
        (y_product_hl >> psimd_splat_u32(16));
    const psimd_u32 z_product_hi = z_abs_hi * vmultiplier_hi +
        (z_product_lh >> psimd_splat_u32(16)) +
        (z_product_hl >> psimd_splat_u32(16));
    const psimd_u32 w_product_hi = w_abs_hi * vmultiplier_hi +
        (w_product_lh >> psimd_splat_u32(16)) +
        (w_product_hl >> psimd_splat_u32(16));

    const psimd_u32 x_adjusted_product = (x_product_hi + vrounding_hi) -
        ((psimd_s32)(x_product_lo & vrounding_lo) >> psimd_splat_s32(31));
    const psimd_u32 y_adjusted_product = (y_product_hi + vrounding_hi) -
        ((psimd_s32)(y_product_lo & vrounding_lo) >> psimd_splat_s32(31));
    const psimd_u32 z_adjusted_product = (z_product_hi + vrounding_hi) -
        ((psimd_s32)(z_product_lo & vrounding_lo) >> psimd_splat_s32(31));
    const psimd_u32 w_adjusted_product = (w_product_hi + vrounding_hi) -
        ((psimd_s32)(w_product_lo & vrounding_lo) >> psimd_splat_s32(31));

    const psimd_u32 x_abs_scaled = x_adjusted_product >> vshift;
    const psimd_u32 y_abs_scaled = y_adjusted_product >> vshift;
    const psimd_u32 z_abs_scaled = z_adjusted_product >> vshift;
    const psimd_u32 w_abs_scaled = w_adjusted_product >> vshift;

    const psimd_s32 x_scaled =
        (psimd_s32)(x_abs_scaled ^ x_neg_mask) - x_neg_mask;
    const psimd_s32 y_scaled =
        (psimd_s32)(y_abs_scaled ^ y_neg_mask) - y_neg_mask;
    const psimd_s32 z_scaled =
        (psimd_s32)(z_abs_scaled ^ z_neg_mask) - z_neg_mask;
    const psimd_s32 w_scaled =
        (psimd_s32)(w_abs_scaled ^ w_neg_mask) - w_neg_mask;

    const psimd_u32 x_clamped =
        (psimd_u32)psimd_max_s32(psimd_min_s32(x_scaled, vsmax), vsmin) +
        vzero_point;
    const psimd_u32 y_clamped =
        (psimd_u32)psimd_max_s32(psimd_min_s32(y_scaled, vsmax), vsmin) +
        vzero_point;
    const psimd_u32 z_clamped =
        (psimd_u32)psimd_max_s32(psimd_min_s32(z_scaled, vsmax), vsmin) +
        vzero_point;
    const psimd_u32 w_clamped =
        (psimd_u32)psimd_max_s32(psimd_min_s32(w_scaled, vsmax), vsmin) +
        vzero_point;

    const psimd_u16 xy_clamped =
        psimd_concat_even_u16((psimd_u16)x_clamped, (psimd_u16)y_clamped);
    const psimd_u16 zw_clamped =
        psimd_concat_even_u16((psimd_u16)z_clamped, (psimd_u16)w_clamped);

    const psimd_u8 xyzw_clamped =
        psimd_concat_even_u8((psimd_u8)xy_clamped, (psimd_u8)zw_clamped);

    psimd_store_u8(output, xyzw_clamped);
    output += 16;
  }
}
