/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <assert.h>

#include <altivec.h>

#include <qnnpack/q8avgpool.h>

void pytorch_q8avgpool_ukernel_up16xm__vsx(
    size_t n,
    size_t ks,
    size_t kc,
    const uint8_t** input,
    const uint8_t* zero,
    uint8_t* output,
    size_t input_increment,
    size_t output_increment,
    const union pytorch_qnnp_avgpool_quantization_params
        quantization_params[RESTRICT_STATIC 1]) {
  assert(n != 0);
  assert(ks != 0);
  assert(kc < 16);

  const vector int vbias = vec_splats(quantization_params->vsx.bias);
  const vector float vscale = vec_splats(quantization_params->vsx.scale);
  const vector short voutput_zero_point =
      vec_splats(quantization_params->vsx.output_zero_point);
  const vector unsigned char vmax =
      vec_splats(quantization_params->vsx.output_max);
  const vector unsigned char vmin =
      vec_splats(quantization_params->vsx.output_min);

  const vector unsigned char vzero = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  const vector unsigned char vshift_2bytes = {
      16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  const vector unsigned char vshift_4bytes = {
      32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  const vector unsigned char vshift_8bytes = {
      64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  // Iterate over the windows
  do {
    const uint8_t** next_input =
        (const uint8_t**)((uintptr_t)input + input_increment);
    // Initialize the accumulators with bias
    vector int vacc_hi_hi = vbias;
    vector int vacc_hi_lo = vbias;
    vector int vacc_lo_hi, vacc_lo_lo;
    if (kc >= 8) {
      vacc_lo_hi = vbias;
      vacc_lo_lo = vbias;
    }

    // Compute one element of the current window per iteration
    size_t m = ks;
    do {
      const uint8_t* i = *input++;
      i += kc;
      vector unsigned char vi = {
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
      // Partial loads are required as the #channels < 16
      if (kc & 1) {
        i -= 1;
        vi = vec_insert(*i, vi, 0);
      }
      if (kc & 2) {
        i -= 2;
        vi = vec_slo(vi, vshift_2bytes);
        vi = (vector unsigned char)vec_insert(
            *(uint16_t*)i, (vector unsigned short)vi, 0);
      }
      if (kc & 4) {
        i -= 4;
        vi = vec_slo(vi, vshift_4bytes);
        vi = (vector unsigned char)vec_insert(
            *(uint32_t*)i, (vector unsigned int)vi, 0);
      }
      if (kc & 8) {
        i -= 8;
        vi = vec_slo(vi, vshift_8bytes);
        vi = (vector unsigned char)vec_insert(
            *(uint64_t*)i, (vector unsigned long long)vi, 0);

        // Accumulate the lower lanes of the vector vi
        const vector short vxi_lo = (vector short)vec_mergel(vi, vzero);
        vacc_lo_hi = vec_add(
            vacc_lo_hi, (vector int)vec_mergeh(vxi_lo, (vector short)vzero));
        vacc_lo_lo = vec_add(
            vacc_lo_lo, (vector int)vec_mergel(vxi_lo, (vector short)vzero));
      }

      // Accumulate the higher lanes of the vector vi
      const vector short vxi_hi = (vector short)vec_mergeh(vi, vzero);
      vacc_hi_hi = vec_add(
          vacc_hi_hi, (vector int)vec_mergeh(vxi_hi, (vector short)vzero));
      vacc_hi_lo = vec_add(
          vacc_hi_lo, (vector int)vec_mergel(vxi_hi, (vector short)vzero));
    } while (--m != 0);
    input = next_input;

    // Multiply the accumulators by a scale and add the output zero point
    const vector float vacc_hi_hi_f = vec_mul(vec_float(vacc_hi_hi), vscale);
    const vector float vacc_hi_lo_f = vec_mul(vec_float(vacc_hi_lo), vscale);

    const vector int vscaled_hi_hi = vec_signed(vec_round(vacc_hi_hi_f));
    const vector int vscaled_hi_lo = vec_signed(vec_round(vacc_hi_lo_f));

    const vector short vout_hi =
        vec_add(vec_packs(vscaled_hi_hi, vscaled_hi_lo), voutput_zero_point);

    // Pack the accumulators into a uint8 vector and check ranges (min/max)
    vector unsigned char vout;
    if (kc > 8) {
      const vector float vacc_lo_hi_f = vec_mul(vec_float(vacc_lo_hi), vscale);
      const vector float vacc_lo_lo_f = vec_mul(vec_float(vacc_lo_lo), vscale);

      const vector int vscaled_lo_hi = vec_signed(vec_round(vacc_lo_hi_f));
      const vector int vscaled_lo_lo = vec_signed(vec_round(vacc_lo_lo_f));

      const vector short vout_lo =
          vec_add(vec_packs(vscaled_lo_hi, vscaled_lo_lo), voutput_zero_point);

      vout = vec_packsu(vout_hi, vout_lo);
    } else {
      vout = vec_packsu(vout_hi, vout_hi);
    }

    vout = vec_min(vout, vmax);
    vout = vec_max(vout, vmin);

    // Partial stores are required as the #channels < 16
    if (kc & 8) {
      *((uint64_t*)output) = ((vector unsigned long long)vout)[0];
      vout = vec_sro(vout, vshift_8bytes);
      output += 8;
    }
    if (kc & 4) {
      *((uint32_t*)output) = ((vector unsigned int)vout)[0];
      vout = vec_sro(vout, vshift_4bytes);
      output += 4;
    }
    if (kc & 2) {
      *((uint16_t*)output) = ((vector unsigned short)vout)[0];
      vout = vec_sro(vout, vshift_2bytes);
      output += 2;
    }
    if (kc & 1) {
      output[0] = vout[0];
      output += 1;
    }

    output = (uint8_t*)((uintptr_t)output + output_increment);
  } while (--n != 0);
}
