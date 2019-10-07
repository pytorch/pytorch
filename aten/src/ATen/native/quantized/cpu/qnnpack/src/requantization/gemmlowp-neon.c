/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <assert.h>
#include <stdint.h>

#include <arm_neon.h>

#include <fp16/bitcasts.h>
#include <qnnpack/requantization-stubs.h>

/*
 * The requantization implementation below is adapted from Google's gemmlowp
 * library. It is only used in QNNPACK unit tests and comparative benchmarks,
 * but not the library itself.
 */

// Copyright 2015 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

void pytorch_qnnp_requantize_gemmlowp__neon(
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

  /* Compute requantization parameters */
  const uint32_t multiplier =
      ((scale_bits & UINT32_C(0x007FFFFF)) | UINT32_C(0x00800000)) << 7;
  const int32_t exponent = (fp32_to_bits(scale) >> 23) - 127 - 23 - 7;
  const int32_t shift =
      -(32 /* using high 32 bits in VQRDMUL */ - 1 /* doubling in VQRDMUL */ +
        exponent);

  const int32x4_t vmultiplier = vdupq_n_s32(multiplier);
  const int16x8_t vzero_point = vdupq_n_s16((int16_t)(uint16_t)zero_point);
  const int32x4_t vshift = vdupq_n_s32(-shift);
  const uint8x16_t vqmin = vdupq_n_u8(qmin);
  const uint8x16_t vqmax = vdupq_n_u8(qmax);
  for (; n != 0; n -= 16) {
    const int32x4_t x = vld1q_s32(input);
    const int32x4_t y = vld1q_s32(input + 4);
    const int32x4_t z = vld1q_s32(input + 8);
    const int32x4_t w = vld1q_s32(input + 12);
    input += 16;

    const int32x4_t x_product = vqrdmulhq_s32(x, vmultiplier);
    const int32x4_t y_product = vqrdmulhq_s32(y, vmultiplier);
    const int32x4_t z_product = vqrdmulhq_s32(z, vmultiplier);
    const int32x4_t w_product = vqrdmulhq_s32(w, vmultiplier);

    const int32x4_t x_product_fixup = vshrq_n_s32(vandq_s32(x, vshift), 31);
    const int32x4_t y_product_fixup = vshrq_n_s32(vandq_s32(y, vshift), 31);
    const int32x4_t z_product_fixup = vshrq_n_s32(vandq_s32(z, vshift), 31);
    const int32x4_t w_product_fixup = vshrq_n_s32(vandq_s32(w, vshift), 31);

    const int32x4_t x_adjusted_product = vqaddq_s32(x_product, x_product_fixup);
    const int32x4_t y_adjusted_product = vqaddq_s32(y_product, y_product_fixup);
    const int32x4_t z_adjusted_product = vqaddq_s32(z_product, z_product_fixup);
    const int32x4_t w_adjusted_product = vqaddq_s32(w_product, w_product_fixup);

    const int32x4_t x_scaled = vrshlq_s32(x_adjusted_product, vshift);
    const int32x4_t y_scaled = vrshlq_s32(y_adjusted_product, vshift);
    const int32x4_t z_scaled = vrshlq_s32(z_adjusted_product, vshift);
    const int32x4_t w_scaled = vrshlq_s32(w_adjusted_product, vshift);

#ifdef __aarch64__
    const int16x8_t xy_packed = vqaddq_s16(
        vqmovn_high_s32(vqmovn_s32(x_scaled), y_scaled), vzero_point);
    const int16x8_t zw_packed = vqaddq_s16(
        vqmovn_high_s32(vqmovn_s32(z_scaled), w_scaled), vzero_point);
    const uint8x16_t xyzw_packed =
        vqmovun_high_s16(vqmovun_s16(xy_packed), zw_packed);
#else
    const int16x8_t xy_packed = vqaddq_s16(
        vcombine_s16(vqmovn_s32(x_scaled), vqmovn_s32(y_scaled)), vzero_point);
    const int16x8_t zw_packed = vqaddq_s16(
        vcombine_s16(vqmovn_s32(z_scaled), vqmovn_s32(w_scaled)), vzero_point);
    const uint8x16_t xyzw_packed =
        vcombine_u8(vqmovun_s16(xy_packed), vqmovun_s16(zw_packed));
#endif

    const uint8x16_t xyzw_clamped =
        vmaxq_u8(vminq_u8(xyzw_packed, vqmax), vqmin);

    vst1q_u8(output, xyzw_clamped);
    output += 16;
  }
}
