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

void pytorch_qnnp_requantize_precise__neon(
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
  const int32_t multiplier =
      ((int32_t)scale_bits & INT32_C(0x007FFFFF)) | INT32_C(0x00800000);
  const int32_t shift = 127 + 23 - (scale_bits >> 23);
  assert(shift >= 24);
  assert(shift < 56);

#if defined(__aarch64__)
  const int32x4_t vmultiplier = vdupq_n_s32(multiplier);
#else
  const int32x2_t vmultiplier = vdup_n_s32(multiplier);
#endif
  const int16x8_t vzero_point = vdupq_n_s16((int16_t)(uint16_t)zero_point);
  const int64x2_t vshift = vdupq_n_s64(-shift);
  const uint8x16_t vqmin = vdupq_n_u8(qmin);
  const uint8x16_t vqmax = vdupq_n_u8(qmax);
  for (; n != 0; n -= 16) {
    const int32x4_t x = vld1q_s32(input);
    const int32x4_t y = vld1q_s32(input + 4);
    const int32x4_t z = vld1q_s32(input + 8);
    const int32x4_t w = vld1q_s32(input + 12);
    input += 16;

    const uint32x4_t x_neg_mask = vcltq_s32(x, vmovq_n_s32(0));
    const uint32x4_t y_neg_mask = vcltq_s32(y, vmovq_n_s32(0));
    const uint32x4_t z_neg_mask = vcltq_s32(z, vmovq_n_s32(0));
    const uint32x4_t w_neg_mask = vcltq_s32(w, vmovq_n_s32(0));

#if defined(__aarch64__)
    const int64x2_t x01_product =
        vmull_s32(vget_low_s32(x), vget_low_s32(vmultiplier));
    const int64x2_t x23_product = vmull_high_s32(x, vmultiplier);
    const int64x2_t y01_product =
        vmull_s32(vget_low_s32(y), vget_low_s32(vmultiplier));
    const int64x2_t y23_product = vmull_high_s32(y, vmultiplier);
    const int64x2_t z01_product =
        vmull_s32(vget_low_s32(z), vget_low_s32(vmultiplier));
    const int64x2_t z23_product = vmull_high_s32(z, vmultiplier);
    const int64x2_t w01_product =
        vmull_s32(vget_low_s32(w), vget_low_s32(vmultiplier));
    const int64x2_t w23_product = vmull_high_s32(w, vmultiplier);
#else
    const int64x2_t x01_product = vmull_s32(vget_low_s32(x), vmultiplier);
    const int64x2_t x23_product = vmull_s32(vget_high_s32(x), vmultiplier);
    const int64x2_t y01_product = vmull_s32(vget_low_s32(y), vmultiplier);
    const int64x2_t y23_product = vmull_s32(vget_high_s32(y), vmultiplier);
    const int64x2_t z01_product = vmull_s32(vget_low_s32(z), vmultiplier);
    const int64x2_t z23_product = vmull_s32(vget_high_s32(z), vmultiplier);
    const int64x2_t w01_product = vmull_s32(vget_low_s32(w), vmultiplier);
    const int64x2_t w23_product = vmull_s32(vget_high_s32(w), vmultiplier);
#endif

#if defined(__aarch64__)
    const int64x2_t x01_adjusted_product =
        vaddw_s32(x01_product, vreinterpret_s32_u32(vget_low_u32(x_neg_mask)));
    const int64x2_t x23_adjusted_product =
        vaddw_high_s32(x23_product, vreinterpretq_s32_u32(x_neg_mask));
    const int64x2_t y01_adjusted_product =
        vaddw_s32(y01_product, vreinterpret_s32_u32(vget_low_u32(y_neg_mask)));
    const int64x2_t y23_adjusted_product =
        vaddw_high_s32(y23_product, vreinterpretq_s32_u32(y_neg_mask));
    const int64x2_t z01_adjusted_product =
        vaddw_s32(z01_product, vreinterpret_s32_u32(vget_low_u32(z_neg_mask)));
    const int64x2_t z23_adjusted_product =
        vaddw_high_s32(z23_product, vreinterpretq_s32_u32(z_neg_mask));
    const int64x2_t w01_adjusted_product =
        vaddw_s32(w01_product, vreinterpret_s32_u32(vget_low_u32(w_neg_mask)));
    const int64x2_t w23_adjusted_product =
        vaddw_high_s32(w23_product, vreinterpretq_s32_u32(w_neg_mask));
#else
    const int64x2_t x01_adjusted_product =
        vaddw_s32(x01_product, vreinterpret_s32_u32(vget_low_u32(x_neg_mask)));
    const int64x2_t x23_adjusted_product =
        vaddw_s32(x23_product, vreinterpret_s32_u32(vget_high_u32(x_neg_mask)));
    const int64x2_t y01_adjusted_product =
        vaddw_s32(y01_product, vreinterpret_s32_u32(vget_low_u32(y_neg_mask)));
    const int64x2_t y23_adjusted_product =
        vaddw_s32(y23_product, vreinterpret_s32_u32(vget_high_u32(y_neg_mask)));
    const int64x2_t z01_adjusted_product =
        vaddw_s32(z01_product, vreinterpret_s32_u32(vget_low_u32(z_neg_mask)));
    const int64x2_t z23_adjusted_product =
        vaddw_s32(z23_product, vreinterpret_s32_u32(vget_high_u32(z_neg_mask)));
    const int64x2_t w01_adjusted_product =
        vaddw_s32(w01_product, vreinterpret_s32_u32(vget_low_u32(w_neg_mask)));
    const int64x2_t w23_adjusted_product =
        vaddw_s32(w23_product, vreinterpret_s32_u32(vget_high_u32(w_neg_mask)));
#endif

    const int64x2_t x01_scaled = vrshlq_s64(x01_adjusted_product, vshift);
    const int64x2_t x23_scaled = vrshlq_s64(x23_adjusted_product, vshift);
    const int64x2_t y01_scaled = vrshlq_s64(y01_adjusted_product, vshift);
    const int64x2_t y23_scaled = vrshlq_s64(y23_adjusted_product, vshift);
    const int64x2_t z01_scaled = vrshlq_s64(z01_adjusted_product, vshift);
    const int64x2_t z23_scaled = vrshlq_s64(z23_adjusted_product, vshift);
    const int64x2_t w01_scaled = vrshlq_s64(w01_adjusted_product, vshift);
    const int64x2_t w23_scaled = vrshlq_s64(w23_adjusted_product, vshift);

#ifdef __aarch64__
    const int32x4_t x_scaled = vuzp1q_s32(
        vreinterpretq_s32_s64(x01_scaled), vreinterpretq_s32_s64(x23_scaled));
    const int32x4_t y_scaled = vuzp1q_s32(
        vreinterpretq_s32_s64(y01_scaled), vreinterpretq_s32_s64(y23_scaled));
    const int32x4_t z_scaled = vuzp1q_s32(
        vreinterpretq_s32_s64(z01_scaled), vreinterpretq_s32_s64(z23_scaled));
    const int32x4_t w_scaled = vuzp1q_s32(
        vreinterpretq_s32_s64(w01_scaled), vreinterpretq_s32_s64(w23_scaled));

    const int16x8_t xy_packed = vqaddq_s16(
        vqmovn_high_s32(vqmovn_s32(x_scaled), y_scaled), vzero_point);
    const int16x8_t zw_packed = vqaddq_s16(
        vqmovn_high_s32(vqmovn_s32(z_scaled), w_scaled), vzero_point);
    const uint8x16_t xyzw_packed =
        vqmovun_high_s16(vqmovun_s16(xy_packed), zw_packed);
#else
    const int32x4_t x_scaled =
        vcombine_s32(vmovn_s64(x01_scaled), vmovn_s64(x23_scaled));
    const int32x4_t y_scaled =
        vcombine_s32(vmovn_s64(y01_scaled), vmovn_s64(y23_scaled));
    const int32x4_t z_scaled =
        vcombine_s32(vmovn_s64(z01_scaled), vmovn_s64(z23_scaled));
    const int32x4_t w_scaled =
        vcombine_s32(vmovn_s64(w01_scaled), vmovn_s64(w23_scaled));

    const int16x8_t xy_packed = vqaddq_s16(
        vcombine_s16(vqmovn_s32(x_scaled), vqmovn_s32(y_scaled)), vzero_point);
    const int16x8_t zw_packed = vqaddq_s16(
        vcombine_s16(vqmovn_s32(z_scaled), vqmovn_s32(w_scaled)), vzero_point);
    const uint8x16_t xyzw_packed =
        vcombine_u8(vqmovun_s16(xy_packed), vqmovun_s16(zw_packed));
#endif

    const uint8x16_t xyzw_clamped =
        vmaxq_u8(vminq_u8(xyzw_packed, vqmax), vqmin);

    /*
     * AArch32 version:
     *   4x VCLT.S32 Qd, Qm, #0
     *   8x VMULL.S32 Qd, Dm, Dn
     *   8x VADDW.S32 Qd, Qm, Dn
     *   8x VRSHL.S32 Qd, Qm, Qn
     *   8x VMOVN.S64 Dd, Qm
     *   4x VQMOVN.S32 Dd, Qm
     *   2x VADD.S16 Qd, Qm, Qn
     *   2x VQMOVUN.S16 Dd, Qm
     *   1x VMAX.U8 Qd, Qm, Qn
     *   1x VMIN.U8 Qd, Qm, Qn
     * ---------------------
     * 46 instructions total
     *
     * AArch64 version:
     *   4x CMLT Vd.4S, Vn.4S, #0
     *   4x SMULL Vd.2D, Vn.2S, Vm.2S
     *   4x SMULL2 Vd.2D, Vn.4S, Vm.4S
     *   4x SADDW Vd.2D, Vn.2D, Vm.2S
     *   4x SADDW2 Vd.2D, Vn.2D, Vm.4S
     *   8x SRSHL Vd.2D, Vn.2D, Vm.2D
     *   4x UZP1 Vd.4S, Vn.4S, Vm.4S
     *   2x SQXTN Vd.4H, Vn.4S
     *   2x SQXTN2 Vd.8H, Vn.4S
     *   2x ADD Vd.8H, Vn.8H, Vm.8H
     *   1x SQXTUN Vd.8B, Vn.8H
     *   1x SQXTUN2 Vd.16B, Vn.8H
     *   1x UMIN Vd.16B, Vn.16B, Vm.16B
     *   1x UMAX Vd.16B, Vn.16B, Vm.16B
     * ---------------------
     * 42 instructions total
     */

    vst1q_u8(output, xyzw_clamped);
    output += 16;
  }
}
