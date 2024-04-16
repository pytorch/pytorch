/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <assert.h>

#include <altivec.h>

#include <qnnpack/u8clamp.h>

void pytorch_u8clamp_ukernel__vsx(
    size_t n,
    const uint8_t* x,
    uint8_t* y,
    const union pytorch_qnnp_u8_clamping_params params[RESTRICT_STATIC 1]) {
  assert(n != 0);

  if
    PYTORCH_QNNP_LIKELY(n >= 8) {
      const vector unsigned char vzero = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
      const vector unsigned char voutput_max =
          vec_splats(params->vsx.output_max);
      const vector unsigned char voutput_min =
          vec_splats(params->vsx.output_min);
      // Iterate over 64 elements
      for (; n >= 64; n -= 64) {
        const vector unsigned char vx0 = vec_xl(0, x);
        const vector unsigned char vx1 = vec_xl(16, x);
        const vector unsigned char vx2 = vec_xl(32, x);
        const vector unsigned char vx3 = vec_xl(48, x);
        x += 64;

        // Clamp the values according to the ranges defined in min/max
        const vector unsigned char vy0 =
            vec_min(vec_max(vx0, voutput_min), voutput_max);
        const vector unsigned char vy1 =
            vec_min(vec_max(vx1, voutput_min), voutput_max);
        const vector unsigned char vy2 =
            vec_min(vec_max(vx2, voutput_min), voutput_max);
        const vector unsigned char vy3 =
            vec_min(vec_max(vx3, voutput_min), voutput_max);

        __builtin_prefetch(x + 640);

        // Store results
        vec_xst(vy0, 0, y);
        vec_xst(vy1, 16, y);
        vec_xst(vy2, 32, y);
        vec_xst(vy3, 48, y);
        y += 64;
      }
      for (; n >= 8; n -= 8) {
        // Compute remaining elements
        vector unsigned char vout = (vector unsigned char)vec_insert(
            ((uint64_t*)x)[0], (vector unsigned long long)vzero, 0);
        x += 8;
        vout = vec_min(vout, voutput_max);
        vout = vec_max(vout, voutput_min);
        *((uint64_t*)y) = vec_extract((vector unsigned long long)vout, 0);
        y += 8;
      }
      if (n != 0) {
        // Compute remaining elements
        const size_t n_increment = n - 8;
        x = (const uint8_t*)((uintptr_t)x + n_increment);
        y = (uint8_t*)((uintptr_t)y + n_increment);

        vector unsigned char vout = (vector unsigned char)vec_insert(
            ((uint64_t*)x)[0], (vector unsigned long long)vzero, 0);
        vout = vec_min(vout, voutput_max);
        vout = vec_max(vout, voutput_min);
        *((uint64_t*)y) = vec_extract((vector unsigned long long)vout, 0);
      }
    }
  else {
    // Scalar execution
    const uint32_t voutput_max = params->vsx.output_max;
    const uint32_t voutput_min = params->vsx.output_min;
    do {
      uint32_t vout = *x++;
      vout = vout > voutput_max ? voutput_max : vout;
      vout = vout < voutput_min ? voutput_min : vout;
      *y++ = (uint8_t)vout;
    } while (--n != 0);
  }
}
