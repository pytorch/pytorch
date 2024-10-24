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

#include <qnnpack/requantization-stubs.h>

void pytorch_qnnp_requantize_fp32__neon(
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

  const float32x4_t vscale = vdupq_n_f32(scale);
#ifdef __aarch64__
  const int16x8_t vzero_point = vdupq_n_s16((int16_t)(uint16_t)zero_point);
  const uint8x16_t vqmin = vdupq_n_u8(qmin);
  const uint8x16_t vqmax = vdupq_n_u8(qmax);
#else
  const float32x4_t vfmin = vdupq_n_f32(
      (float)((int32_t)(uint32_t)qmin - (int32_t)(uint32_t)zero_point));
  const float32x4_t vfmax = vdupq_n_f32(
      (float)((int32_t)(uint32_t)qmax - (int32_t)(uint32_t)zero_point));
  const float32x4_t vfmagic = vdupq_n_f32(12582912.0f);
  const int32x4_t vimagic =
      vdupq_n_s32(INT32_C(0x4B400000) - (int32_t)(uint32_t)zero_point);
#endif
  for (; n != 0; n -= 16) {
    const int32x4_t x = vld1q_s32(input);
    const int32x4_t y = vld1q_s32(input + 4);
    const int32x4_t z = vld1q_s32(input + 8);
    const int32x4_t w = vld1q_s32(input + 12);
    input += 16;

    /*
     * Convert int32_t input to FP32 and multiply by FP32 scale.
     * Both operations involve statistically unbiased roundings:
     * - Large int32_t values can't be exactly represented as FP32. The
     * conversion instruction in ARM NEON would round it to nearest FP32 value
     * with ties to even.
     * - Product of two FP32 values is generally not exactly representation as
     * an FP32 value, and will be rounded to nearest FP32 value with ties to
     * even.
     */
    const float32x4_t x_scaled = vmulq_f32(vcvtq_f32_s32(x), vscale);
    const float32x4_t y_scaled = vmulq_f32(vcvtq_f32_s32(y), vscale);
    const float32x4_t z_scaled = vmulq_f32(vcvtq_f32_s32(z), vscale);
    const float32x4_t w_scaled = vmulq_f32(vcvtq_f32_s32(w), vscale);

#ifdef __aarch64__
    /*
     * Leverage "Floating-point Convert to Signed integer, rounding to nearest
     * with ties to even" instruction. This is an ARMv8 instruction (always
     * available in AArch64), which saturates result on overflow. We don't need
     * to specifically consider saturated results, they will be clamped at the
     * last stage.
     */
    const int32x4_t x_rounded = vcvtnq_s32_f32(x_scaled);
    const int32x4_t y_rounded = vcvtnq_s32_f32(y_scaled);
    const int32x4_t z_rounded = vcvtnq_s32_f32(z_scaled);
    const int32x4_t w_rounded = vcvtnq_s32_f32(w_scaled);

    /*
     * Standard final sequence on ARM NEON:
     * - Pack to int16_t and saturate
     * - Add zero point
     * - Pack to uint8_t and saturate
     * - Clamp between qmin and qmax
     */
    const int16x8_t xy_packed = vqaddq_s16(
        vqmovn_high_s32(vqmovn_s32(x_rounded), y_rounded), vzero_point);
    const int16x8_t zw_packed = vqaddq_s16(
        vqmovn_high_s32(vqmovn_s32(z_rounded), w_rounded), vzero_point);
    const uint8x16_t xyzw_packed =
        vqmovun_high_s16(vqmovun_s16(xy_packed), zw_packed);
    const uint8x16_t xyzw_clamped =
        vmaxq_u8(vminq_u8(xyzw_packed, vqmax), vqmin);

    vst1q_u8(output, xyzw_clamped);
    output += 16;
#else
    /*
     * ARMv7 NEON offers only a floating-point to integer conversion instruction
     * with rounding towards zero. In lieu of conversion instruction with
     * rounding-to-nearest-even, we use a magic trick of adding a large number
     * (1.5 * 2**23) to scaled value to cause rounding to integer, and then
     * substracing this magic number as integer. This trick works only in a
     * limited range (absolute value of input must be less than 2**22), so
     * generally we have to clamp input to this range before using the magic.
     * However, clamping to any smaller range works just as well, and thus we
     * clamp to [qmin - zero point, qmax - zero point] range so that after we
     * add zero point to the result, it gets into target [qmin, qmax] range.
     */
    const float32x4_t x_clamped = vminq_f32(vmaxq_f32(x_scaled, vfmin), vfmax);
    const float32x4_t y_clamped = vminq_f32(vmaxq_f32(y_scaled, vfmin), vfmax);
    const float32x4_t z_clamped = vminq_f32(vmaxq_f32(z_scaled, vfmin), vfmax);
    const float32x4_t w_clamped = vminq_f32(vmaxq_f32(w_scaled, vfmin), vfmax);

    /*
     * Conversion to integer using the "magic trick". Rounding is performed in
     * the output of addition operation, and result is rounded to nearest even
     * integer with ties to even.
     */
    const int32x4_t x_biased = vsubq_s32(
        vreinterpretq_s32_f32(vaddq_f32(x_clamped, vfmagic)), vimagic);
    const int32x4_t y_biased = vsubq_s32(
        vreinterpretq_s32_f32(vaddq_f32(y_clamped, vfmagic)), vimagic);
    const int32x4_t z_biased = vsubq_s32(
        vreinterpretq_s32_f32(vaddq_f32(z_clamped, vfmagic)), vimagic);
    const int32x4_t w_biased = vsubq_s32(
        vreinterpretq_s32_f32(vaddq_f32(w_clamped, vfmagic)), vimagic);

    /*
     * Select low 8 bits of each 32-bit integer in the vectors for the output.
     * Since result is already clamped to [qmin, qmax] subrange of [0, 255],
     * saturation is not needed.
     */
    const int16x8_t xy_packed =
        vcombine_s16(vmovn_s32(x_biased), vmovn_s32(y_biased));
    const int16x8_t zw_packed =
        vcombine_s16(vmovn_s32(z_biased), vmovn_s32(w_biased));
    const uint8x16_t xyzw_packed = vreinterpretq_u8_s8(
        vcombine_s8(vmovn_s16(xy_packed), vmovn_s16(zw_packed)));

    /*
     * AArch32 version:
     *   4x VCVT.F32.S32 Qd, Qm
     *   4x VMUL.F32 Qd, Qm, Qn
     *   4x VMIN.F32 Qd, Qm, Qn
     *   4x VMAX.F32 Qd, Qm, Qn
     *   4x VADD.F32 Qd, Qm, Qn
     *   4x VSUB.S32 Qd, Qm, Qn
     *   4x VMOVN.I32 Dd, Qm
     *   2x VMOVN.I16 Dd, Qm
     * ---------------------
     * 30 instructions total
     *
     * AArch64 version:
     *   4x SCVTF Vd.4S, Vn.4S
     *   4x FMUL Vd.4S, Vn.4S, Vm.4S
     *   4x FCVTNS Vd.4S, Vn.4S
     *   2x SQXTN Vd.4H, Vn.4S
     *   2x SQXTN2 Vd.8H, Vn.4S
     *   2x ADD Vd.8H, Vn.8H, Vm.8H
     *   1x SQXTUN Vd.8B, Vn.8H
     *   1x SQXTUN2 Vd.16B, Vn.8H
     *   1x UMIN Vd.16B, Vn.16B, Vm.16B
     *   1x UMAX Vd.16B, Vn.16B, Vm.16B
     * ---------------------
     * 22 instructions total
     */

    vst1q_u8(output, xyzw_packed);
    output += 16;
#endif
  }
}
