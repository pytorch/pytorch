/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <assert.h>

#include <altivec.h>

#include <qnnpack/q8gavgpool.h>

void pytorch_q8gavgpool_ukernel_mp16x7p7q__vsx(
    size_t m,
    size_t n,
    const uint8_t* input,
    size_t input_stride,
    const uint8_t* zero,
    int32_t* buffer,
    uint8_t* output,
    const union pytorch_qnnp_avgpool_quantization_params
        quantization_params[RESTRICT_STATIC 1]) {
  assert(m > 7);
  assert(n >= 16);

  const uint8_t* i0 = input;
  const uint8_t* i1 = i0 + input_stride;
  const uint8_t* i2 = i1 + input_stride;
  const uint8_t* i3 = i2 + input_stride;
  const uint8_t* i4 = i3 + input_stride;
  const uint8_t* i5 = i4 + input_stride;
  const uint8_t* i6 = i5 + input_stride;
  const size_t packed_n = (n + 15) & -16;
  const size_t input_increment = 7 * input_stride - packed_n;

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

  // Iterate over 16 channels and accumulate 7 elements per iteration
  // note: goes up to 15 elements over bound
  int32_t* acc = buffer;
  for (size_t k = 0; k < n; k += 16) {
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

    // Convert the lower and higher int8 lanes into int16 vectors
    const vector short vxi0_hi = (vector short)vec_mergeh(vi0, vzero);
    const vector short vxi0_lo = (vector short)vec_mergel(vi0, vzero);
    const vector short vxi1_hi = (vector short)vec_mergeh(vi1, vzero);
    const vector short vxi1_lo = (vector short)vec_mergel(vi1, vzero);
    const vector short vxi2_hi = (vector short)vec_mergeh(vi2, vzero);
    const vector short vxi2_lo = (vector short)vec_mergel(vi2, vzero);
    const vector short vxi3_hi = (vector short)vec_mergeh(vi3, vzero);
    const vector short vxi3_lo = (vector short)vec_mergel(vi3, vzero);
    const vector short vxi4_hi = (vector short)vec_mergeh(vi4, vzero);
    const vector short vxi4_lo = (vector short)vec_mergel(vi4, vzero);
    const vector short vxi5_hi = (vector short)vec_mergeh(vi5, vzero);
    const vector short vxi5_lo = (vector short)vec_mergel(vi5, vzero);
    const vector short vxi6_hi = (vector short)vec_mergeh(vi6, vzero);
    const vector short vxi6_lo = (vector short)vec_mergel(vi6, vzero);

    // Accumulate the input of each channel and add bias
    vector int vacc_hi_hi = vec_add(vbias, vec_unpackh(vxi0_hi));
    vector int vacc_hi_lo = vec_add(vbias, vec_unpackl(vxi0_hi));
    vector int vacc_lo_hi = vec_add(vbias, vec_unpackh(vxi0_lo));
    vector int vacc_lo_lo = vec_add(vbias, vec_unpackl(vxi0_lo));
    vacc_hi_hi = vec_add(vacc_hi_hi, vec_unpackh(vxi1_hi));
    vacc_hi_lo = vec_add(vacc_hi_lo, vec_unpackl(vxi1_hi));
    vacc_lo_hi = vec_add(vacc_lo_hi, vec_unpackh(vxi1_lo));
    vacc_lo_lo = vec_add(vacc_lo_lo, vec_unpackl(vxi1_lo));
    vacc_hi_hi = vec_add(vacc_hi_hi, vec_unpackh(vxi2_hi));
    vacc_hi_lo = vec_add(vacc_hi_lo, vec_unpackl(vxi2_hi));
    vacc_lo_hi = vec_add(vacc_lo_hi, vec_unpackh(vxi2_lo));
    vacc_lo_lo = vec_add(vacc_lo_lo, vec_unpackl(vxi2_lo));
    vacc_hi_hi = vec_add(vacc_hi_hi, vec_unpackh(vxi3_hi));
    vacc_hi_lo = vec_add(vacc_hi_lo, vec_unpackl(vxi3_hi));
    vacc_lo_hi = vec_add(vacc_lo_hi, vec_unpackh(vxi3_lo));
    vacc_lo_lo = vec_add(vacc_lo_lo, vec_unpackl(vxi3_lo));
    vacc_hi_hi = vec_add(vacc_hi_hi, vec_unpackh(vxi4_hi));
    vacc_hi_lo = vec_add(vacc_hi_lo, vec_unpackl(vxi4_hi));
    vacc_lo_hi = vec_add(vacc_lo_hi, vec_unpackh(vxi4_lo));
    vacc_lo_lo = vec_add(vacc_lo_lo, vec_unpackl(vxi4_lo));
    vacc_hi_hi = vec_add(vacc_hi_hi, vec_unpackh(vxi5_hi));
    vacc_hi_lo = vec_add(vacc_hi_lo, vec_unpackl(vxi5_hi));
    vacc_lo_hi = vec_add(vacc_lo_hi, vec_unpackh(vxi5_lo));
    vacc_lo_lo = vec_add(vacc_lo_lo, vec_unpackl(vxi5_lo));
    vacc_hi_hi = vec_add(vacc_hi_hi, vec_unpackh(vxi6_hi));
    vacc_hi_lo = vec_add(vacc_hi_lo, vec_unpackl(vxi6_hi));
    vacc_lo_hi = vec_add(vacc_lo_hi, vec_unpackh(vxi6_lo));
    vacc_lo_lo = vec_add(vacc_lo_lo, vec_unpackl(vxi6_lo));

    // Store the partial results of the accumulators
    vec_xst(vacc_hi_hi, 0, acc);
    vec_xst(vacc_hi_lo, 16, acc);
    vec_xst(vacc_lo_hi, 32, acc);
    vec_xst(vacc_lo_lo, 48, acc);
    acc += 16;
  }

  // Continue computing the next window elements, 7 per iterations
  for (m -= 7; m > 7; m -= 7) {
    acc = buffer;
    i0 = (const uint8_t*)((uintptr_t)i0 + input_increment);
    i1 = (const uint8_t*)((uintptr_t)i1 + input_increment);
    i2 = (const uint8_t*)((uintptr_t)i2 + input_increment);
    i3 = (const uint8_t*)((uintptr_t)i3 + input_increment);
    i4 = (const uint8_t*)((uintptr_t)i4 + input_increment);
    i5 = (const uint8_t*)((uintptr_t)i5 + input_increment);
    i6 = (const uint8_t*)((uintptr_t)i6 + input_increment);

    // note: goes up to 15 elements over bound
    for (size_t k = 0; k < n; k += 16) {
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

      // Load the partial result to continue accumulating on them
      vector int vacc_hi_hi = vec_xl(0, acc);
      vector int vacc_hi_lo = vec_xl(16, acc);
      vector int vacc_lo_hi = vec_xl(32, acc);
      vector int vacc_lo_lo = vec_xl(48, acc);

      const vector short vxi0_hi = (vector short)vec_mergeh(vi0, vzero);
      const vector short vxi0_lo = (vector short)vec_mergel(vi0, vzero);
      const vector short vxi1_hi = (vector short)vec_mergeh(vi1, vzero);
      const vector short vxi1_lo = (vector short)vec_mergel(vi1, vzero);
      const vector short vxi2_hi = (vector short)vec_mergeh(vi2, vzero);
      const vector short vxi2_lo = (vector short)vec_mergel(vi2, vzero);
      const vector short vxi3_hi = (vector short)vec_mergeh(vi3, vzero);
      const vector short vxi3_lo = (vector short)vec_mergel(vi3, vzero);
      const vector short vxi4_hi = (vector short)vec_mergeh(vi4, vzero);
      const vector short vxi4_lo = (vector short)vec_mergel(vi4, vzero);
      const vector short vxi5_hi = (vector short)vec_mergeh(vi5, vzero);
      const vector short vxi5_lo = (vector short)vec_mergel(vi5, vzero);
      const vector short vxi6_hi = (vector short)vec_mergeh(vi6, vzero);
      const vector short vxi6_lo = (vector short)vec_mergel(vi6, vzero);

      vacc_hi_hi = vec_add(vacc_hi_hi, vec_unpackh(vxi0_hi));
      vacc_hi_lo = vec_add(vacc_hi_lo, vec_unpackl(vxi0_hi));
      vacc_lo_hi = vec_add(vacc_lo_hi, vec_unpackh(vxi0_lo));
      vacc_lo_lo = vec_add(vacc_lo_lo, vec_unpackl(vxi0_lo));
      vacc_hi_hi = vec_add(vacc_hi_hi, vec_unpackh(vxi1_hi));
      vacc_hi_lo = vec_add(vacc_hi_lo, vec_unpackl(vxi1_hi));
      vacc_lo_hi = vec_add(vacc_lo_hi, vec_unpackh(vxi1_lo));
      vacc_lo_lo = vec_add(vacc_lo_lo, vec_unpackl(vxi1_lo));
      vacc_hi_hi = vec_add(vacc_hi_hi, vec_unpackh(vxi2_hi));
      vacc_hi_lo = vec_add(vacc_hi_lo, vec_unpackl(vxi2_hi));
      vacc_lo_hi = vec_add(vacc_lo_hi, vec_unpackh(vxi2_lo));
      vacc_lo_lo = vec_add(vacc_lo_lo, vec_unpackl(vxi2_lo));
      vacc_hi_hi = vec_add(vacc_hi_hi, vec_unpackh(vxi3_hi));
      vacc_hi_lo = vec_add(vacc_hi_lo, vec_unpackl(vxi3_hi));
      vacc_lo_hi = vec_add(vacc_lo_hi, vec_unpackh(vxi3_lo));
      vacc_lo_lo = vec_add(vacc_lo_lo, vec_unpackl(vxi3_lo));
      vacc_hi_hi = vec_add(vacc_hi_hi, vec_unpackh(vxi4_hi));
      vacc_hi_lo = vec_add(vacc_hi_lo, vec_unpackl(vxi4_hi));
      vacc_lo_hi = vec_add(vacc_lo_hi, vec_unpackh(vxi4_lo));
      vacc_lo_lo = vec_add(vacc_lo_lo, vec_unpackl(vxi4_lo));
      vacc_hi_hi = vec_add(vacc_hi_hi, vec_unpackh(vxi5_hi));
      vacc_hi_lo = vec_add(vacc_hi_lo, vec_unpackl(vxi5_hi));
      vacc_lo_hi = vec_add(vacc_lo_hi, vec_unpackh(vxi5_lo));
      vacc_lo_lo = vec_add(vacc_lo_lo, vec_unpackl(vxi5_lo));
      vacc_hi_hi = vec_add(vacc_hi_hi, vec_unpackh(vxi6_hi));
      vacc_hi_lo = vec_add(vacc_hi_lo, vec_unpackl(vxi6_hi));
      vacc_lo_hi = vec_add(vacc_lo_hi, vec_unpackh(vxi6_lo));
      vacc_lo_lo = vec_add(vacc_lo_lo, vec_unpackl(vxi6_lo));

      // Store the partial results of the accumulators
      vec_xst(vacc_hi_hi, 0, acc);
      vec_xst(vacc_hi_lo, 16, acc);
      vec_xst(vacc_lo_hi, 32, acc);
      vec_xst(vacc_lo_lo, 48, acc);
      acc += 16;
    }
  }

  /* Compute the remaining elements of the window. If there are no elements
   * left to compute, zeros are loaded instead */
  i0 = (const uint8_t*)((uintptr_t)i0 + input_increment);
  i1 = (const uint8_t*)((uintptr_t)i1 + input_increment);
  if (m < 2) {
    i1 = zero;
  }
  i2 = (const uint8_t*)((uintptr_t)i2 + input_increment);
  if (m <= 2) {
    i2 = zero;
  }
  i3 = (const uint8_t*)((uintptr_t)i3 + input_increment);
  if (m < 4) {
    i3 = zero;
  }
  i4 = (const uint8_t*)((uintptr_t)i4 + input_increment);
  if (m <= 4) {
    i4 = zero;
  }
  i5 = (const uint8_t*)((uintptr_t)i5 + input_increment);
  if (m < 6) {
    i5 = zero;
  }
  i6 = (const uint8_t*)((uintptr_t)i6 + input_increment);
  if (m <= 6) {
    i6 = zero;
  }

  acc = buffer;
  do {
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

    // Load into the accumulators the partial results
    vector int vacc_hi_hi = vec_xl(0, acc);
    vector int vacc_hi_lo = vec_xl(16, acc);
    vector int vacc_lo_hi = vec_xl(32, acc);
    vector int vacc_lo_lo = vec_xl(48, acc);
    acc += 16;

    const vector short vxi0_hi = (vector short)vec_mergeh(vi0, vzero);
    const vector short vxi0_lo = (vector short)vec_mergel(vi0, vzero);
    const vector short vxi1_hi = (vector short)vec_mergeh(vi1, vzero);
    const vector short vxi1_lo = (vector short)vec_mergel(vi1, vzero);
    const vector short vxi2_hi = (vector short)vec_mergeh(vi2, vzero);
    const vector short vxi2_lo = (vector short)vec_mergel(vi2, vzero);
    const vector short vxi3_hi = (vector short)vec_mergeh(vi3, vzero);
    const vector short vxi3_lo = (vector short)vec_mergel(vi3, vzero);
    const vector short vxi4_hi = (vector short)vec_mergeh(vi4, vzero);
    const vector short vxi4_lo = (vector short)vec_mergel(vi4, vzero);
    const vector short vxi5_hi = (vector short)vec_mergeh(vi5, vzero);
    const vector short vxi5_lo = (vector short)vec_mergel(vi5, vzero);
    const vector short vxi6_hi = (vector short)vec_mergeh(vi6, vzero);
    const vector short vxi6_lo = (vector short)vec_mergel(vi6, vzero);

    // Accumulate the last elements
    vacc_hi_hi = vec_add(vacc_hi_hi, vec_unpackh(vxi0_hi));
    vacc_hi_lo = vec_add(vacc_hi_lo, vec_unpackl(vxi0_hi));
    vacc_lo_hi = vec_add(vacc_lo_hi, vec_unpackh(vxi0_lo));
    vacc_lo_lo = vec_add(vacc_lo_lo, vec_unpackl(vxi0_lo));
    vacc_hi_hi = vec_add(vacc_hi_hi, vec_unpackh(vxi1_hi));
    vacc_hi_lo = vec_add(vacc_hi_lo, vec_unpackl(vxi1_hi));
    vacc_lo_hi = vec_add(vacc_lo_hi, vec_unpackh(vxi1_lo));
    vacc_lo_lo = vec_add(vacc_lo_lo, vec_unpackl(vxi1_lo));
    vacc_hi_hi = vec_add(vacc_hi_hi, vec_unpackh(vxi2_hi));
    vacc_hi_lo = vec_add(vacc_hi_lo, vec_unpackl(vxi2_hi));
    vacc_lo_hi = vec_add(vacc_lo_hi, vec_unpackh(vxi2_lo));
    vacc_lo_lo = vec_add(vacc_lo_lo, vec_unpackl(vxi2_lo));
    vacc_hi_hi = vec_add(vacc_hi_hi, vec_unpackh(vxi3_hi));
    vacc_hi_lo = vec_add(vacc_hi_lo, vec_unpackl(vxi3_hi));
    vacc_lo_hi = vec_add(vacc_lo_hi, vec_unpackh(vxi3_lo));
    vacc_lo_lo = vec_add(vacc_lo_lo, vec_unpackl(vxi3_lo));
    vacc_hi_hi = vec_add(vacc_hi_hi, vec_unpackh(vxi4_hi));
    vacc_hi_lo = vec_add(vacc_hi_lo, vec_unpackl(vxi4_hi));
    vacc_lo_hi = vec_add(vacc_lo_hi, vec_unpackh(vxi4_lo));
    vacc_lo_lo = vec_add(vacc_lo_lo, vec_unpackl(vxi4_lo));
    vacc_hi_hi = vec_add(vacc_hi_hi, vec_unpackh(vxi5_hi));
    vacc_hi_lo = vec_add(vacc_hi_lo, vec_unpackl(vxi5_hi));
    vacc_lo_hi = vec_add(vacc_lo_hi, vec_unpackh(vxi5_lo));
    vacc_lo_lo = vec_add(vacc_lo_lo, vec_unpackl(vxi5_lo));
    vacc_hi_hi = vec_add(vacc_hi_hi, vec_unpackh(vxi6_hi));
    vacc_hi_lo = vec_add(vacc_hi_lo, vec_unpackl(vxi6_hi));
    vacc_lo_hi = vec_add(vacc_lo_hi, vec_unpackh(vxi6_lo));
    vacc_lo_lo = vec_add(vacc_lo_lo, vec_unpackl(vxi6_lo));

    // Multiply the accumulators by a scale and add the output zero point
    const vector float vacc_hi_hi_f = vec_mul(vec_float(vacc_hi_hi), vscale);
    const vector float vacc_hi_lo_f = vec_mul(vec_float(vacc_hi_lo), vscale);
    const vector float vacc_lo_hi_f = vec_mul(vec_float(vacc_lo_hi), vscale);
    const vector float vacc_lo_lo_f = vec_mul(vec_float(vacc_lo_lo), vscale);

    const vector int vscaled_hi_hi = vec_signed(vec_round(vacc_hi_hi_f));
    const vector int vscaled_hi_lo = vec_signed(vec_round(vacc_hi_lo_f));
    const vector int vscaled_lo_hi = vec_signed(vec_round(vacc_lo_hi_f));
    const vector int vscaled_lo_lo = vec_signed(vec_round(vacc_lo_lo_f));

    // Pack the accumulators into a uint8 vector and check ranges (min/max)
    const vector short vout_hi =
        vec_add(vec_packs(vscaled_hi_hi, vscaled_hi_lo), voutput_zero_point);
    const vector short vout_lo =
        vec_add(vec_packs(vscaled_lo_hi, vscaled_lo_lo), voutput_zero_point);

    vector unsigned char vout = vec_packsu(vout_hi, vout_lo);
    vout = vec_min(vout, vmax);
    vout = vec_max(vout, vmin);

    // Store the final results from the accumulators
    vec_xst(vout, 0, output);
    output += 16;

    n -= 16;
  } while (n >= 16);
  if (n != 0) {
    // Compute the remaining channels
    const size_t address_decrement = 16 - n;
    i0 = (const uint8_t*)((uintptr_t)i0 - address_decrement);
    i1 = (const uint8_t*)((uintptr_t)i1 - address_decrement);
    i2 = (const uint8_t*)((uintptr_t)i2 - address_decrement);
    i3 = (const uint8_t*)((uintptr_t)i3 - address_decrement);
    i4 = (const uint8_t*)((uintptr_t)i4 - address_decrement);
    i5 = (const uint8_t*)((uintptr_t)i5 - address_decrement);
    i6 = (const uint8_t*)((uintptr_t)i6 - address_decrement);

    // Shift the elements zeroing the lower lanes
    const vector unsigned char vshift = {
        8 * address_decrement, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    const vector unsigned char vi0 = vec_sro(vec_xl(0, i0), vshift);
    const vector unsigned char vi1 = vec_sro(vec_xl(0, i1), vshift);
    const vector unsigned char vi2 = vec_sro(vec_xl(0, i2), vshift);
    const vector unsigned char vi3 = vec_sro(vec_xl(0, i3), vshift);
    const vector unsigned char vi4 = vec_sro(vec_xl(0, i4), vshift);
    const vector unsigned char vi5 = vec_sro(vec_xl(0, i5), vshift);
    const vector unsigned char vi6 = vec_sro(vec_xl(0, i6), vshift);

    vector int vacc_hi_hi = vec_xl(0, acc);
    vector int vacc_hi_lo = vec_xl(16, acc);

    const vector short vxi0_hi = (vector short)vec_mergeh(vi0, vzero);
    const vector short vxi1_hi = (vector short)vec_mergeh(vi1, vzero);
    const vector short vxi2_hi = (vector short)vec_mergeh(vi2, vzero);
    const vector short vxi3_hi = (vector short)vec_mergeh(vi3, vzero);
    const vector short vxi4_hi = (vector short)vec_mergeh(vi4, vzero);
    const vector short vxi5_hi = (vector short)vec_mergeh(vi5, vzero);
    const vector short vxi6_hi = (vector short)vec_mergeh(vi6, vzero);

    vacc_hi_hi = vec_add(vacc_hi_hi, vec_unpackh(vxi0_hi));
    vacc_hi_lo = vec_add(vacc_hi_lo, vec_unpackl(vxi0_hi));
    vacc_hi_hi = vec_add(vacc_hi_hi, vec_unpackh(vxi1_hi));
    vacc_hi_lo = vec_add(vacc_hi_lo, vec_unpackl(vxi1_hi));
    vacc_hi_hi = vec_add(vacc_hi_hi, vec_unpackh(vxi2_hi));
    vacc_hi_lo = vec_add(vacc_hi_lo, vec_unpackl(vxi2_hi));
    vacc_hi_hi = vec_add(vacc_hi_hi, vec_unpackh(vxi3_hi));
    vacc_hi_lo = vec_add(vacc_hi_lo, vec_unpackl(vxi3_hi));
    vacc_hi_hi = vec_add(vacc_hi_hi, vec_unpackh(vxi4_hi));
    vacc_hi_lo = vec_add(vacc_hi_lo, vec_unpackl(vxi4_hi));
    vacc_hi_hi = vec_add(vacc_hi_hi, vec_unpackh(vxi5_hi));
    vacc_hi_lo = vec_add(vacc_hi_lo, vec_unpackl(vxi5_hi));
    vacc_hi_hi = vec_add(vacc_hi_hi, vec_unpackh(vxi6_hi));
    vacc_hi_lo = vec_add(vacc_hi_lo, vec_unpackl(vxi6_hi));

    const vector float vacc_hi_hi_f = vec_mul(vec_float(vacc_hi_hi), vscale);
    const vector float vacc_hi_lo_f = vec_mul(vec_float(vacc_hi_lo), vscale);

    const vector int vscaled_hi_hi = vec_signed(vec_round(vacc_hi_hi_f));
    const vector int vscaled_hi_lo = vec_signed(vec_round(vacc_hi_lo_f));

    const vector short vout_hi =
        vec_add(vec_packs(vscaled_hi_hi, vscaled_hi_lo), voutput_zero_point);

    vector unsigned char vout;
    if (n > 8) {
      const vector short vxi0_lo = (vector short)vec_mergel(vi0, vzero);
      const vector short vxi1_lo = (vector short)vec_mergel(vi1, vzero);
      const vector short vxi2_lo = (vector short)vec_mergel(vi2, vzero);
      const vector short vxi3_lo = (vector short)vec_mergel(vi3, vzero);
      const vector short vxi4_lo = (vector short)vec_mergel(vi4, vzero);
      const vector short vxi5_lo = (vector short)vec_mergel(vi5, vzero);
      const vector short vxi6_lo = (vector short)vec_mergel(vi6, vzero);

      vector int vacc_lo_hi = vec_xl(32, acc);
      vector int vacc_lo_lo = vec_xl(48, acc);

      vacc_lo_hi = vec_add(vacc_lo_hi, vec_unpackh(vxi0_lo));
      vacc_lo_lo = vec_add(vacc_lo_lo, vec_unpackl(vxi0_lo));
      vacc_lo_hi = vec_add(vacc_lo_hi, vec_unpackh(vxi1_lo));
      vacc_lo_lo = vec_add(vacc_lo_lo, vec_unpackl(vxi1_lo));
      vacc_lo_hi = vec_add(vacc_lo_hi, vec_unpackh(vxi2_lo));
      vacc_lo_lo = vec_add(vacc_lo_lo, vec_unpackl(vxi2_lo));
      vacc_lo_hi = vec_add(vacc_lo_hi, vec_unpackh(vxi3_lo));
      vacc_lo_lo = vec_add(vacc_lo_lo, vec_unpackl(vxi3_lo));
      vacc_lo_hi = vec_add(vacc_lo_hi, vec_unpackh(vxi4_lo));
      vacc_lo_lo = vec_add(vacc_lo_lo, vec_unpackl(vxi4_lo));
      vacc_lo_hi = vec_add(vacc_lo_hi, vec_unpackh(vxi5_lo));
      vacc_lo_lo = vec_add(vacc_lo_lo, vec_unpackl(vxi5_lo));
      vacc_lo_hi = vec_add(vacc_lo_hi, vec_unpackh(vxi6_lo));
      vacc_lo_lo = vec_add(vacc_lo_lo, vec_unpackl(vxi6_lo));

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

    // Store the results generated by the remaining channels
    if (n & 8) {
      *(uint64_t*)output = ((vector unsigned long long)vout)[0];
      output += 8;
      const vector unsigned char vshift_8bytes = {
          8 * 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
      vout = vec_sro(vout, vshift_8bytes);
    }
    if (n & 4) {
      *(uint32_t*)output = ((vector unsigned int)vout)[0];
      output += 4;
      const vector unsigned char vshift_4bytes = {
          8 * 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
      vout = vec_sro(vout, vshift_4bytes);
    }
    if (n & 2) {
      *(uint16_t*)output = ((vector unsigned short)vout)[0];
      output += 2;
      const vector unsigned char vshift_2bytes = {
          8 * 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
      vout = vec_sro(vout, vshift_2bytes);
    }
    if (n & 1) {
      output[0] = vout[0];
      output += 1;
    }
  }
}
