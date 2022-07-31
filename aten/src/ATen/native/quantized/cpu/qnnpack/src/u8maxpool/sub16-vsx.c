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

void pytorch_u8maxpool_ukernel_sub16__vsx(
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
  assert(kc != 0);
  assert(kc < 16);

  const vector unsigned char voutput_max = vec_splats(params->vsx.output_max);
  const vector unsigned char voutput_min = vec_splats(params->vsx.output_min);

  const vector unsigned char vzero = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  const vector unsigned char mask_shift_2bytes = {
      16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  const vector unsigned char mask_shift_4bytes = {
      32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  const vector unsigned char mask_shift_8bytes = {
      64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  // Iterate over the windows
  do {
    vector unsigned char vmax =
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    // Compute one element of the current window per iteration
    size_t m = ks;
    do {
      const uint8_t* i = *input++;
      i += kc;
      vector unsigned char vi = vmax;

      // Partial loads are required as the #channels < 16
      if (kc & 1) {
        i -= 1;
        vi = vec_insert(*i, vi, 0);
      }
      if (kc & 2) {
        i -= 2;
        vi = vec_slo(vi, mask_shift_2bytes);
        vi = (vector unsigned char)vec_insert(
            *(uint16_t *)i, (vector unsigned short)vi, 0);
      }
      if (kc & 4) {
        i -= 4;
        vi = vec_slo(vi, mask_shift_4bytes);
        vi = (vector unsigned char)vec_insert(
            *(uint32_t *)i, (vector unsigned int)vi, 0);
      }
      if (kc & 8) {
        i -= 8;
        vi = vec_slo(vi, mask_shift_8bytes);
        vi = (vector unsigned char)vec_insert(
            *(uint64_t *)i, (vector unsigned long long)vi, 0);
      }

      // Compute maximum over all loaded channels
      vmax = vec_max(vmax, vi);
    } while (--m != 0);
    input = (const uint8_t**)((uintptr_t)input + input_increment);
    vector unsigned char vout =
        vec_max(vec_min(vmax, voutput_max), voutput_min);

    // Partial stores are required as the #channels < 16
    if (kc & 8) {
      *((uint64_t *)output) = ((vector unsigned long long)vout)[0];
      vout = vec_sro(vout, mask_shift_8bytes);
      output += 8;
    }
    if (kc & 4) {
      *((uint32_t *)output) = ((vector unsigned int)vout)[0];
      vout = vec_sro(vout, mask_shift_4bytes);
      output += 4;
    }
    if (kc & 2) {
      *((uint16_t *)output) = ((vector unsigned short)vout)[0];
      vout = vec_sro(vout, mask_shift_2bytes);
      output += 2;
    }
    if (kc & 1) {
      output[0] = vout[0];
      output += 1;
    }

    output = (uint8_t*)((uintptr_t)output + output_increment);
  } while (--n != 0);
}
