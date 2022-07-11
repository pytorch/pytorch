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

#include <qnnpack/requantization-stubs.h>

void pytorch_qnnp_requantize_fp32__psimd(
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

  const psimd_f32 vscale = psimd_splat_f32(scale);
  const psimd_f32 vfmin = psimd_splat_f32(
      (float)((int32_t)(uint32_t)qmin - (int32_t)(uint32_t)zero_point));
  const psimd_f32 vfmax = psimd_splat_f32(
      (float)((int32_t)(uint32_t)qmax - (int32_t)(uint32_t)zero_point));
  const psimd_f32 vfmagic = psimd_splat_f32(12582912.0f);
  const psimd_s32 vimagic =
      psimd_splat_s32(INT32_C(0x4B400000) - (int32_t)(uint32_t)zero_point);
  for (; n != 0; n -= 16) {
    const psimd_s32 x = psimd_load_s32(input);
    const psimd_s32 y = psimd_load_s32(input + 4);
    const psimd_s32 z = psimd_load_s32(input + 8);
    const psimd_s32 w = psimd_load_s32(input + 12);
    input += 16;

    /*
     * Convert int32_t input to FP32 and multiply by FP32 scale.
     * Both operations involve roundings:
     * - Large int32_t values can't be exactly represented as FP32. We expect
     * that conversion instruction would round it to nearest FP32 value with
     * ties to even, but Clang documentation for __builtin_convertvector does
     *   not guarantee that.
     * - Product of two FP32 values is generally not exactly representation as
     * an FP32 value, and will be rounded to nearest FP32 value with ties to
     * even.
     */
    const psimd_f32 x_scaled = psimd_cvt_s32_f32(x) * vscale;
    const psimd_f32 y_scaled = psimd_cvt_s32_f32(y) * vscale;
    const psimd_f32 z_scaled = psimd_cvt_s32_f32(z) * vscale;
    const psimd_f32 w_scaled = psimd_cvt_s32_f32(w) * vscale;

    /*
     * Clang/gcc vector extension does not provide an intrinsics for a
     * floating-point to integer conversion operation with
     * rounding-to-nearest-even. In lieu of such intrinsic, we use a magic trick
     * of adding a large number (1.5 * 2**23) to scaled value to cause rounding
     * to integer, and then substracing this magic number as integer. This trick
     * works only in a limited range (absolute value of input must be less than
     * 2**22), so generally we have to clamp input to this range before using
     * the magic. However, clamping to any smaller range works just as well, and
     * thus we clamp to [qmin - zero point, qmax - zero point] range so that
     * after we add zero point to the result, it gets into target [qmin, qmax]
     * range.
     */
    const psimd_f32 x_clamped =
        psimd_min_f32(psimd_max_f32(x_scaled, vfmin), vfmax);
    const psimd_f32 y_clamped =
        psimd_min_f32(psimd_max_f32(y_scaled, vfmin), vfmax);
    const psimd_f32 z_clamped =
        psimd_min_f32(psimd_max_f32(z_scaled, vfmin), vfmax);
    const psimd_f32 w_clamped =
        psimd_min_f32(psimd_max_f32(w_scaled, vfmin), vfmax);

    /*
     * Conversion to integer using the "magic trick". Rounding is performed in
     * the output of addition operation, and result is rounded to nearest even
     * integer with ties to even.
     */
    const psimd_s32 x_biased = (psimd_s32)(x_clamped + vfmagic) - vimagic;
    const psimd_s32 y_biased = (psimd_s32)(y_clamped + vfmagic) - vimagic;
    const psimd_s32 z_biased = (psimd_s32)(z_clamped + vfmagic) - vimagic;
    const psimd_s32 w_biased = (psimd_s32)(w_clamped + vfmagic) - vimagic;

    /*
     * Select low 8 bits of each 32-bit integer in the vectors for the output.
     * Since result is already clamped to [qmin, qmax] subrange of [0, 255],
     * saturation is not needed.
     */
    const psimd_u16 xy_packed =
        psimd_concat_even_u16((psimd_u16)x_biased, (psimd_u16)y_biased);
    const psimd_u16 zw_packed =
        psimd_concat_even_u16((psimd_u16)z_biased, (psimd_u16)w_biased);

    const psimd_u8 xyzw_packed =
        psimd_concat_even_u8((psimd_u8)xy_packed, (psimd_u8)zw_packed);

    psimd_store_u8(output, xyzw_packed);
    output += 16;
  }
}
