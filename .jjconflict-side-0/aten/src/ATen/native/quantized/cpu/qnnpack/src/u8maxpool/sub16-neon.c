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

void pytorch_u8maxpool_ukernel_sub16__neon(
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
  assert(kc != 0);
  assert(kc < 16);

  const uint8x16_t voutput_max = vld1q_dup_u8(&params->neon.output_max);
  const uint8x16_t voutput_min = vld1q_dup_u8(&params->neon.output_min);
  do {
    uint8x16_t vmax = vmovq_n_u8(0);

    size_t m = ks;
    do {
      const uint8_t* i = *input++;
      i += kc;
      uint8x16_t vi = vmax;
      if (kc & 1) {
        i -= 1;
        vi = vld1q_lane_u8(i, vi, 0);
      }
      if (kc & 2) {
        vi = vextq_u8(vi, vi, 14);
        i -= 2;
        vi = vreinterpretq_u8_u16(vld1q_lane_u16(
            __builtin_assume_aligned(i, 1), vreinterpretq_u16_u8(vi), 0));
      }
      if (kc & 4) {
        vi = vextq_u8(vi, vi, 12);
        i -= 4;
        vi = vreinterpretq_u8_u32(vld1q_lane_u32(
            __builtin_assume_aligned(i, 1), vreinterpretq_u32_u8(vi), 0));
      }
      if (kc & 8) {
        i -= 8;
        vi = vcombine_u8(vld1_u8(i), vget_low_u8(vi));
      }
      vmax = vmaxq_u8(vmax, vi);
    } while (--m != 0);
    input = (const uint8_t**)((uintptr_t)input + input_increment);

    vmax = vminq_u8(vmax, voutput_max);
    vmax = vmaxq_u8(vmax, voutput_min);

    uint8x8_t vout = vget_low_u8(vmax);
    if (kc & 8) {
      vst1_u8(output, vout);
      output += 8;
      vout = vget_high_u8(vmax);
    }
    if (kc & 4) {
      vst1_lane_u32(
          __builtin_assume_aligned(output, 1), vreinterpret_u32_u8(vout), 0);
      output += 4;
      vout = vext_u8(vout, vout, 4);
    }
    if (kc & 2) {
      vst1_lane_u16(
          __builtin_assume_aligned(output, 1), vreinterpret_u16_u8(vout), 0);
      output += 2;
      vout = vext_u8(vout, vout, 2);
    }
    if (kc & 1) {
      vst1_lane_u8(output, vout, 0);
      output += 1;
    }
    output = (uint8_t*)((uintptr_t)output + output_increment);

  } while (--n != 0);
}
