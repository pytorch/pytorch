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

#include "gemmlowp-scalar.h"

void pytorch_qnnp_requantize_gemmlowp__scalar(
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

  /* Compute requantization parameters */
  const uint32_t multiplier =
      ((scale_bits & UINT32_C(0x007FFFFF)) | UINT32_C(0x00800000)) << 7;
  const int32_t exponent = (fp32_to_bits(scale) >> 23) - 127 - 23 - 7;
  const int32_t shift =
      -(32 /* using high 32 bits in VQRDMUL */ - 1 /* doubling in VQRDMUL */ +
        exponent);

  const int32_t smin = (int32_t)(uint32_t)qmin;
  const int32_t smax = (int32_t)(uint32_t)qmax;
  for (; n != 0; n -= 4) {
    const int32_t x = input[0];
    const int32_t y = input[1];
    const int32_t z = input[2];
    const int32_t w = input[3];
    input += 4;

    const int32_t x_product = gemmlowp_scalar_vqrdmulh_s32(x, multiplier);
    const int32_t y_product = gemmlowp_scalar_vqrdmulh_s32(y, multiplier);
    const int32_t z_product = gemmlowp_scalar_vqrdmulh_s32(z, multiplier);
    const int32_t w_product = gemmlowp_scalar_vqrdmulh_s32(w, multiplier);

    const int32_t x_scaled = gemmlowp_scalar_rdivbypo2_s32(x_product, shift);
    const int32_t y_scaled = gemmlowp_scalar_rdivbypo2_s32(y_product, shift);
    const int32_t z_scaled = gemmlowp_scalar_rdivbypo2_s32(z_product, shift);
    const int32_t w_scaled = gemmlowp_scalar_rdivbypo2_s32(w_product, shift);

    /* Add zero point to scaled value */
    const int32_t x_biased = x_scaled + zero_point;
    const int32_t y_biased = y_scaled + zero_point;
    const int32_t z_biased = z_scaled + zero_point;
    const int32_t w_biased = w_scaled + zero_point;

    /* Clamp scaled value with zero point between smin and smax */
    const int32_t x_clamped =
        x_biased < smin ? smin : x_biased > smax ? smax : x_biased;
    const int32_t y_clamped =
        y_biased < smin ? smin : y_biased > smax ? smax : y_biased;
    const int32_t z_clamped =
        z_biased < smin ? smin : z_biased > smax ? smax : z_biased;
    const int32_t w_clamped =
        w_biased < smin ? smin : w_biased > smax ? smax : w_biased;

    output[0] = (uint8_t)x_clamped;
    output[1] = (uint8_t)y_clamped;
    output[2] = (uint8_t)z_clamped;
    output[3] = (uint8_t)w_clamped;
    output += 4;
  }
}
