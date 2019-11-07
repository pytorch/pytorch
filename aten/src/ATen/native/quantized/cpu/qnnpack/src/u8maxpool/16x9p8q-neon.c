/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <assert.h>

#include <arm_neon.h>

#include <qnnpack/u8maxpool.h>

void pytorch_u8maxpool_ukernel_16x9p8q__neon(
    size_t n,
    size_t ks,
    size_t kc,
    const uint8_t** input,
    uint8_t* output,
    size_t input_increment,
    size_t output_increment,
    const union pytorch_qnnp_u8_clamping_params params[restrict static 1]) {
  assert(n != 0);
  assert(ks != 0);
  assert(kc >= 16);

  const uint8x16_t voutput_max = vld1q_dup_u8(&params->neon.output_max);
  const uint8x16_t voutput_min = vld1q_dup_u8(&params->neon.output_min);
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

      size_t k = kc;
      while (k >= 16) {
        const uint8x16_t vi0 = vld1q_u8(i0);
        i0 += 16;
        const uint8x16_t vi1 = vld1q_u8(i1);
        i1 += 16;
        const uint8x16_t vi2 = vld1q_u8(i2);
        i2 += 16;
        const uint8x16_t vi3 = vld1q_u8(i3);
        i3 += 16;
        const uint8x16_t vi4 = vld1q_u8(i4);
        i4 += 16;
        const uint8x16_t vi5 = vld1q_u8(i5);
        i5 += 16;
        const uint8x16_t vi6 = vld1q_u8(i6);
        i6 += 16;
        const uint8x16_t vi7 = vld1q_u8(i7);
        i7 += 16;
        const uint8x16_t vi8 = vld1q_u8(i8);
        i8 += 16;

        const uint8x16_t vmax018 = vmaxq_u8(vmaxq_u8(vi0, vi1), vi8);
        const uint8x16_t vmax23 = vmaxq_u8(vi2, vi3);
        const uint8x16_t vmax45 = vmaxq_u8(vi4, vi5);
        const uint8x16_t vmax67 = vmaxq_u8(vi6, vi7);

        const uint8x16_t vmax2345 = vmaxq_u8(vmax23, vmax45);
        const uint8x16_t vmax01678 = vmaxq_u8(vmax018, vmax67);
        const uint8x16_t vmax = vmaxq_u8(vmax2345, vmax01678);
        const uint8x16_t vout =
            vmaxq_u8(vminq_u8(vmax, voutput_max), voutput_min);

        vst1q_u8(o, vout);
        o += 16;

        k -= 16;
      }
      if (k != 0) {
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

        const uint8x16_t vi0 = vld1q_u8(i0);
        const uint8x16_t vi1 = vld1q_u8(i1);
        const uint8x16_t vi2 = vld1q_u8(i2);
        const uint8x16_t vi3 = vld1q_u8(i3);
        const uint8x16_t vi4 = vld1q_u8(i4);
        const uint8x16_t vi5 = vld1q_u8(i5);
        const uint8x16_t vi6 = vld1q_u8(i6);
        const uint8x16_t vi7 = vld1q_u8(i7);
        const uint8x16_t vi8 = vld1q_u8(i8);

        const uint8x16_t vmax018 = vmaxq_u8(vmaxq_u8(vi0, vi1), vi8);
        const uint8x16_t vmax23 = vmaxq_u8(vi2, vi3);
        const uint8x16_t vmax45 = vmaxq_u8(vi4, vi5);
        const uint8x16_t vmax67 = vmaxq_u8(vi6, vi7);

        const uint8x16_t vmax2345 = vmaxq_u8(vmax23, vmax45);
        const uint8x16_t vmax01678 = vmaxq_u8(vmax018, vmax67);
        const uint8x16_t vmax = vmaxq_u8(vmax2345, vmax01678);
        const uint8x16_t vout =
            vmaxq_u8(vminq_u8(vmax, voutput_max), voutput_min);

        vst1q_u8(o, vout);
        o += 16;
      }
    }

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
        const uint8x16_t vi0 = vld1q_u8(i0);
        i0 += 16;
        const uint8x16_t vi1 = vld1q_u8(i1);
        i1 += 16;
        const uint8x16_t vi2 = vld1q_u8(i2);
        i2 += 16;
        const uint8x16_t vi3 = vld1q_u8(i3);
        i3 += 16;
        const uint8x16_t vi4 = vld1q_u8(i4);
        i4 += 16;
        const uint8x16_t vi5 = vld1q_u8(i5);
        i5 += 16;
        const uint8x16_t vi6 = vld1q_u8(i6);
        i6 += 16;
        const uint8x16_t vi7 = vld1q_u8(i7);
        i7 += 16;
        const uint8x16_t vo = vld1q_u8(o);

        const uint8x16_t vmax01 = vmaxq_u8(vmaxq_u8(vi0, vi1), vo);
        const uint8x16_t vmax23 = vmaxq_u8(vi2, vi3);
        const uint8x16_t vmax45 = vmaxq_u8(vi4, vi5);
        const uint8x16_t vmax67 = vmaxq_u8(vi6, vi7);

        const uint8x16_t vmax2345 = vmaxq_u8(vmax23, vmax45);
        const uint8x16_t vmax0167 = vmaxq_u8(vmax01, vmax67);
        const uint8x16_t vmax = vmaxq_u8(vmax2345, vmax0167);
        const uint8x16_t vout =
            vmaxq_u8(vminq_u8(vmax, voutput_max), voutput_min);

        vst1q_u8(o, vout);
        o += 16;

        k -= 16;
      }
      if (k != 0) {
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

        const uint8x16_t vi0 = vld1q_u8(i0);
        const uint8x16_t vi1 = vld1q_u8(i1);
        const uint8x16_t vi2 = vld1q_u8(i2);
        const uint8x16_t vi3 = vld1q_u8(i3);
        const uint8x16_t vi4 = vld1q_u8(i4);
        const uint8x16_t vi5 = vld1q_u8(i5);
        const uint8x16_t vi6 = vld1q_u8(i6);
        const uint8x16_t vi7 = vld1q_u8(i7);
        const uint8x16_t vo = vld1q_u8(o);

        const uint8x16_t vmax01 = vmaxq_u8(vmaxq_u8(vi0, vi1), vo);
        const uint8x16_t vmax23 = vmaxq_u8(vi2, vi3);
        const uint8x16_t vmax45 = vmaxq_u8(vi4, vi5);
        const uint8x16_t vmax67 = vmaxq_u8(vi6, vi7);

        const uint8x16_t vmax2345 = vmaxq_u8(vmax23, vmax45);
        const uint8x16_t vmax0167 = vmaxq_u8(vmax01, vmax67);
        const uint8x16_t vmax = vmaxq_u8(vmax2345, vmax0167);
        const uint8x16_t vout =
            vmaxq_u8(vminq_u8(vmax, voutput_max), voutput_min);

        vst1q_u8(o, vout);
        o += 16;
      }
    }
    input = (const uint8_t**)((uintptr_t)input + input_increment);
    output = (uint8_t*)((uintptr_t)o + output_increment);
  } while (--n != 0);
}
