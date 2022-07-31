/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <assert.h>

#include <altivec.h>

#include <qnnpack/u8maxpool.h>

void pytorch_u8maxpool_ukernel_16x9p8q__vsx(
    size_t n,
    size_t ks,
    size_t kc,
    const uint8_t** input,
    uint8_t* output,
    size_t input_increment,
    size_t output_increment,
    const union pytorch_qnnp_u8_clamping_params params[RESTRICT_STATIC 1]) {
  assert(n != 0);
  assert(ks != 0);
  assert(kc >= 16);

  const vector unsigned char voutput_max = vec_splats(params->vsx.output_max);
  const vector unsigned char voutput_min = vec_splats(params->vsx.output_min);

  const vector unsigned char vzero = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  // Iterate over the windows
  do {
    uint8_t* o = output;
    {
      const uint8_t* i0 = *input++;
      const uint8_t* i1 = *input++;
      const uint8_t* i2 = *input++;
      const uint8_t* i3 = *input++;
      const uint8_t* i4 = *input++;
      const uint8_t* i5 = *input++;
      const uint8_t* i6 = *input++;
      const uint8_t* i7 = *input++;
      const uint8_t* i8 = *input++;
      if (ks < 2) {
        i1 = i0;
      }
      if (ks <= 2) {
        i2 = i0;
      }
      if (ks < 4) {
        i3 = i0;
      }
      if (ks <= 4) {
        i4 = i0;
      }
      if (ks < 6) {
        i5 = i0;
      }
      if (ks <= 6) {
        i6 = i0;
      }
      if (ks < 8) {
        i7 = i0;
      }
      if (ks <= 8) {
        i8 = i0;
      }

      // Iterate over 16 channels computing up to 9 elements per iteration
      size_t k = kc;
      while (k >= 16) {
        const vector unsigned char vi0 = vec_xl(0, i0);
        i0 += 16;
        const vector unsigned char vi1 = vec_xl(0, i1);
        i1 += 16;
        const vector unsigned char vi2 = vec_xl(0, i2);
        i2 += 16;
        const vector unsigned char vi3 = vec_xl(0, i3);
        i3 += 16;
        const vector unsigned char vi4 = vec_xl(0, i4);
        i4 += 16;
        const vector unsigned char vi5 = vec_xl(0, i5);
        i5 += 16;
        const vector unsigned char vi6 = vec_xl(0, i6);
        i6 += 16;
        const vector unsigned char vi7 = vec_xl(0, i7);
        i7 += 16;
        const vector unsigned char vi8 = vec_xl(0, i8);
        i8 += 16;

        // Identify the maximum element of each channel
        const vector unsigned char vmax018 = vec_max(vec_max(vi0, vi1), vi8);
        const vector unsigned char vmax23 = vec_max(vi2, vi3);
        const vector unsigned char vmax45 = vec_max(vi4, vi5);
        const vector unsigned char vmax67 = vec_max(vi6, vi7);

        const vector unsigned char vmax2345 = vec_max(vmax23, vmax45);
        const vector unsigned char vmax01678 = vec_max(vmax018, vmax67);
        const vector unsigned char vmax = vec_max(vmax2345, vmax01678);

        // Check ranges (min/max) and store the partial result
        const vector unsigned char vout =
            vec_max(vec_min(vmax, voutput_max), voutput_min);

        vec_xst(vout, 0, o);
        o += 16;

        k -= 16;
      }
      if (k != 0) {
        // Compute the remaining channels
        const size_t address_increment = k - 16;
        i0 = (const uint8_t*)((uintptr_t)i0 + address_increment);
        i1 = (const uint8_t*)((uintptr_t)i1 + address_increment);
        i2 = (const uint8_t*)((uintptr_t)i2 + address_increment);
        i3 = (const uint8_t*)((uintptr_t)i3 + address_increment);
        i4 = (const uint8_t*)((uintptr_t)i4 + address_increment);
        i5 = (const uint8_t*)((uintptr_t)i5 + address_increment);
        i6 = (const uint8_t*)((uintptr_t)i6 + address_increment);
        i7 = (const uint8_t*)((uintptr_t)i7 + address_increment);
        i8 = (const uint8_t*)((uintptr_t)i8 + address_increment);
        o = (uint8_t*)((uintptr_t)o + address_increment);

        const vector unsigned char vi0 = vec_xl(0, i0);
        const vector unsigned char vi1 = vec_xl(0, i1);
        const vector unsigned char vi2 = vec_xl(0, i2);
        const vector unsigned char vi3 = vec_xl(0, i3);
        const vector unsigned char vi4 = vec_xl(0, i4);
        const vector unsigned char vi5 = vec_xl(0, i5);
        const vector unsigned char vi6 = vec_xl(0, i6);
        const vector unsigned char vi7 = vec_xl(0, i7);
        const vector unsigned char vi8 = vec_xl(0, i8);

        const vector unsigned char vmax018 = vec_max(vec_max(vi0, vi1), vi8);
        const vector unsigned char vmax23 = vec_max(vi2, vi3);
        const vector unsigned char vmax45 = vec_max(vi4, vi5);
        const vector unsigned char vmax67 = vec_max(vi6, vi7);

        const vector unsigned char vmax2345 = vec_max(vmax23, vmax45);
        const vector unsigned char vmax01678 = vec_max(vmax018, vmax67);
        const vector unsigned char vmax = vec_max(vmax2345, vmax01678);
        const vector unsigned char vout =
            vec_max(vec_min(vmax, voutput_max), voutput_min);

        vec_xst(vout, 0, o);
        o += 16;
      }
    }

    /* Continue computing the next valid elements of the windows -- up to 8
     * elements per iterations */
    for (ptrdiff_t m = (ptrdiff_t)ks - 9; m > 0; m -= 8) {
      const uint8_t* i0 = *input++;
      const uint8_t* i1 = *input++;
      const uint8_t* i2 = *input++;
      const uint8_t* i3 = *input++;
      const uint8_t* i4 = *input++;
      const uint8_t* i5 = *input++;
      const uint8_t* i6 = *input++;
      const uint8_t* i7 = *input++;
      if (m < 2) {
        i1 = i0;
      }
      if (m <= 2) {
        i2 = i0;
      }
      if (m < 4) {
        i3 = i0;
      }
      if (m <= 4) {
        i4 = i0;
      }
      if (m < 6) {
        i5 = i0;
      }
      if (m <= 6) {
        i6 = i0;
      }
      if (m < 8) {
        i7 = i0;
      }

      o = output;
      size_t k = kc;
      while (k >= 16) {
        const vector unsigned char vi0 = vec_xl(0, i0);
        i0 += 16;
        const vector unsigned char vi1 = vec_xl(0, i1);
        i1 += 16;
        const vector unsigned char vi2 = vec_xl(0, i2);
        i2 += 16;
        const vector unsigned char vi3 = vec_xl(0, i3);
        i3 += 16;
        const vector unsigned char vi4 = vec_xl(0, i4);
        i4 += 16;
        const vector unsigned char vi5 = vec_xl(0, i5);
        i5 += 16;
        const vector unsigned char vi6 = vec_xl(0, i6);
        i6 += 16;
        const vector unsigned char vi7 = vec_xl(0, i7);
        i7 += 16;

        // Load partial result into vector vo
        const vector unsigned char vo = vec_xl(0, o);

        const vector unsigned char vmax01 = vec_max(vec_max(vi0, vi1), vo);
        const vector unsigned char vmax23 = vec_max(vi2, vi3);
        const vector unsigned char vmax45 = vec_max(vi4, vi5);
        const vector unsigned char vmax67 = vec_max(vi6, vi7);

        const vector unsigned char vmax2345 = vec_max(vmax23, vmax45);
        const vector unsigned char vmax0167 = vec_max(vmax01, vmax67);
        const vector unsigned char vmax = vec_max(vmax2345, vmax0167);
        const vector unsigned char vout =
            vec_max(vec_min(vmax, voutput_max), voutput_min);

        // Store final result
        vec_xst(vout, 0, o);
        o += 16;

        k -= 16;
      }
      if (k != 0) {
        // Compute remaining channels
        const size_t address_increment = k - 16;
        i0 = (const uint8_t*)((uintptr_t)i0 + address_increment);
        i1 = (const uint8_t*)((uintptr_t)i1 + address_increment);
        i2 = (const uint8_t*)((uintptr_t)i2 + address_increment);
        i3 = (const uint8_t*)((uintptr_t)i3 + address_increment);
        i4 = (const uint8_t*)((uintptr_t)i4 + address_increment);
        i5 = (const uint8_t*)((uintptr_t)i5 + address_increment);
        i6 = (const uint8_t*)((uintptr_t)i6 + address_increment);
        i7 = (const uint8_t*)((uintptr_t)i7 + address_increment);
        o = (uint8_t*)((uintptr_t)o + address_increment);

        const vector unsigned char vi0 = vec_xl(0, i0);
        const vector unsigned char vi1 = vec_xl(0, i1);
        const vector unsigned char vi2 = vec_xl(0, i2);
        const vector unsigned char vi3 = vec_xl(0, i3);
        const vector unsigned char vi4 = vec_xl(0, i4);
        const vector unsigned char vi5 = vec_xl(0, i5);
        const vector unsigned char vi6 = vec_xl(0, i6);
        const vector unsigned char vi7 = vec_xl(0, i7);
        const vector unsigned char vo = vec_xl(0, o);

        const vector unsigned char vmax01 = vec_max(vec_max(vi0, vi1), vo);
        const vector unsigned char vmax23 = vec_max(vi2, vi3);
        const vector unsigned char vmax45 = vec_max(vi4, vi5);
        const vector unsigned char vmax67 = vec_max(vi6, vi7);

        const vector unsigned char vmax2345 = vec_max(vmax23, vmax45);
        const vector unsigned char vmax0167 = vec_max(vmax01, vmax67);
        const vector unsigned char vmax = vec_max(vmax2345, vmax0167);
        const vector unsigned char vout =
            vec_max(vec_min(vmax, voutput_max), voutput_min);
        vec_xst(vout, 0, o);
        o += 16;
      }
    }
    input = (const uint8_t**)((uintptr_t)input + input_increment);
    output = (uint8_t*)((uintptr_t)o + output_increment);
  } while (--n != 0);
}
