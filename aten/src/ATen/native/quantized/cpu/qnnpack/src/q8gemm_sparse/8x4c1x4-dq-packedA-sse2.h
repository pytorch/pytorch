/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <immintrin.h>

#include <qnnpack/q8gemm_sparse.h>
#include <requantization/runtime-sse2.h>

#include "8x4c1x4-packed-sse2.h"

#define CONVERT_TO_FP_AND_TRANSPOSE(a, b, c, d, t_a, t_b, t_c, t_d)  \
  a_ps = _mm_cvtepi32_ps(a);                                         \
  b_ps = _mm_cvtepi32_ps(b);                                         \
  c_ps = _mm_cvtepi32_ps(c);                                         \
  d_ps = _mm_cvtepi32_ps(d);                                         \
  tmp0 = _mm_shuffle_ps(a_ps, b_ps, _MM_SHUFFLE(1, 0, 1, 0));        \
  tmp1 = _mm_shuffle_ps(a_ps, b_ps, _MM_SHUFFLE(3, 2, 3, 2));        \
  tmp2 = _mm_shuffle_ps(c_ps, d_ps, _MM_SHUFFLE(1, 0, 1, 0));        \
  tmp3 = _mm_shuffle_ps(c_ps, d_ps, _MM_SHUFFLE(3, 2, 3, 2));        \
  t_a = _mm_shuffle_ps(tmp0, tmp2, _MM_SHUFFLE(2, 0, 2, 0));         \
  t_b = _mm_shuffle_ps(tmp0, tmp2, _MM_SHUFFLE(3, 1, 3, 1));         \
  t_c = _mm_shuffle_ps(tmp1, tmp3, _MM_SHUFFLE(2, 0, 2, 0));         \
  t_d = _mm_shuffle_ps(tmp1, tmp3, _MM_SHUFFLE(3, 1, 3, 1));

// KERNEL_NAME and W_INDEX_DTYPE macros are defined in
// https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/quantized/cpu/qnnpack/src/q8gemm_sparse/8x4c1x4-dq-packedA-sse2.c
void KERNEL_NAME(
    size_t mr,
    size_t nr,
    const uint8_t* a_packed,
    const uint8_t* packed_w,
    const W_INDEX_DTYPE* w_row_ptr,
    const W_INDEX_DTYPE* w_block_ids_ptr,
    const float* b,
    float* c,
    size_t c_stride,
    size_t output_channel_index,
    const struct pytorch_qnnp_conv_dynamic_quantization_params
        quantization_params[RESTRICT_STATIC 1]) {
  const __m128i va_zero_point = _mm_set1_epi16(quantization_params->input_zero_point);
  const __m128 vbias = _mm_load_ps(b);
  const __m128i vzero = _mm_setzero_si128();

  // Packed A format.
  // 8kx4m blocks for alls blocks given 4 rows (4m) are placed in contiguous memory.
  // Original A
  // --------- K -----------          -- (K + 4 - 1) / 4 --
  // |                     |          |                   |
  // |                     |        (M + 8 - 1)/8         |
  // |                     | Packed   |                   |
  // M                     |  =>      |-------------------|
  // |                     |        Thus Packed A has (K + 4 - 1)/4 * (M + 8 -1)/8 blocks
  // |                     |
  // |---------------------|
  //
  // Each 8 x 4 blocks is transposed and stored.
  // Each of the (K + 4 - 1)/4 blocks for a given group of 8 m blocks
  // are stored adjacent in memory
  // Thus, each block:
  // |----8m-----|----8m-----|
  // 4k          |           | .....
  // |-----------|-----------|
  // This locality helps in loading 8kx8m blocks of activations
  // Note when M is not multiple of 8, the rest can contain arbitrary
  // data in packed A as we will not be writing those out.
  // This wil be taken care by just copying the appropriate valid data

  __m128i vacc_low[4];
  __m128i vacc_high[4];
  const __m128 vmultiplier =
      _mm_loadu_ps(&quantization_params->multipliers[output_channel_index]);
  for (int32_t n = 0; n < nr; n++) {
    vacc_low[n] = _mm_setzero_si128();
    vacc_high[n] = _mm_setzero_si128();
    const int16_t b_zero_point =
      (int16_t)(uint16_t)quantization_params->kernel_zero_points[
      output_channel_index + n];

    int32_t num_blocks = w_row_ptr[n+1] - w_row_ptr[n];
    // Offset into compressed values.
    // w_row_ptr[0] is the block offset in the compressed values.
    // Where the corresponding row of the weight matrix starts.
    const uint8_t* temp_packed_w = packed_w + w_row_ptr[n] * COL_BLOCK_SIZE;
    // Similarly w_row_ptr[0] is also the block offset where
    // corresponding row's block column ids start.
    // Per row # of block column ids = # of block values
    const W_INDEX_DTYPE* temp_w_block_ids_ptr = w_block_ids_ptr + w_row_ptr[n];
    while (num_blocks > 1) {
      // Load two 1x4 uint8 blocks 2 ints
      const uint8_t* b_ptr = temp_packed_w;
      // This is not perf optimal since this will result in
      // register spills. We probably should work with output block
      // of 1x4 instead of 1x8
      // But doing is this way because mostly this how we will
      // do it for ARM and this reference code helps establish
      // the baseline for functional correctness.
      const int16_t b_0 = (int16_t)((uint16_t)(b_ptr[0]));
      const int16_t b_1 = (int16_t)((uint16_t)(b_ptr[1]));
      const int16_t b_2 = (int16_t)((uint16_t)(b_ptr[2]));
      const int16_t b_3 = (int16_t)((uint16_t)(b_ptr[3]));
      const int16_t b_4 = (int16_t)((uint16_t)(b_ptr[4]));
      const int16_t b_5 = (int16_t)((uint16_t)(b_ptr[5]));
      const int16_t b_6 = (int16_t)((uint16_t)(b_ptr[6]));
      const int16_t b_7 = (int16_t)((uint16_t)(b_ptr[7]));
      // Now we will load 8kx1(broadcast 8) weight values
      const __m128i vxb0 = _mm_set1_epi16((b_0 - b_zero_point));
      const __m128i vxb1 = _mm_set1_epi16((b_1 - b_zero_point));
      const __m128i vxb2 = _mm_set1_epi16((b_2 - b_zero_point));
      const __m128i vxb3 = _mm_set1_epi16((b_3 - b_zero_point));
      const __m128i vxb4 = _mm_set1_epi16((b_4 - b_zero_point));
      const __m128i vxb5 = _mm_set1_epi16((b_5 - b_zero_point));
      const __m128i vxb6 = _mm_set1_epi16((b_6 - b_zero_point));
      const __m128i vxb7 = _mm_set1_epi16((b_7 - b_zero_point));

      // Load activation blocks. In this kernel we assume
      // a mat is already transposed. K x M
      // 1. Load 8 1x8 registers = 8k x 8m

      // Load column id of the first 1x4 block
      int32_t col_block_id_0 = temp_w_block_ids_ptr[0];
      // Load column id of the second 1x4 block
      int32_t col_block_id_1 = temp_w_block_ids_ptr[1];
      const __m128i va0 =
        _mm_loadl_epi64((const __m128i*) (a_packed +
            col_block_id_0 * PACKED_A_BLOCK_SIZE + MR * 0));
      const __m128i va1 =
        _mm_loadl_epi64((const __m128i*) (a_packed +
            col_block_id_0 * PACKED_A_BLOCK_SIZE + MR * 1));
      const __m128i va2 =
        _mm_loadl_epi64((const __m128i*) (a_packed +
            col_block_id_0 * PACKED_A_BLOCK_SIZE + MR * 2));
      const __m128i va3 =
        _mm_loadl_epi64((const __m128i*) (a_packed +
            col_block_id_0 * PACKED_A_BLOCK_SIZE + MR * 3));
      const __m128i va4 =
        _mm_loadl_epi64((const __m128i*) (a_packed +
            col_block_id_1 * PACKED_A_BLOCK_SIZE + MR * 0));
      const __m128i va5 =
        _mm_loadl_epi64((const __m128i*) (a_packed +
            col_block_id_1 * PACKED_A_BLOCK_SIZE + MR * 1));
      const __m128i va6 =
        _mm_loadl_epi64((const __m128i*) (a_packed +
            col_block_id_1 * PACKED_A_BLOCK_SIZE + MR * 2));
      const __m128i va7 =
        _mm_loadl_epi64((const __m128i*) (a_packed +
            col_block_id_1 * PACKED_A_BLOCK_SIZE + MR * 3));

      const __m128i vxa0 =
          sub_zero_point(_mm_unpacklo_epi8(va0, vzero), va_zero_point);
      const __m128i vxa1 =
          sub_zero_point(_mm_unpacklo_epi8(va1, vzero), va_zero_point);
      const __m128i vxa2 =
          sub_zero_point(_mm_unpacklo_epi8(va2, vzero), va_zero_point);
      const __m128i vxa3 =
          sub_zero_point(_mm_unpacklo_epi8(va3, vzero), va_zero_point);
      const __m128i vxa4 =
          sub_zero_point(_mm_unpacklo_epi8(va4, vzero), va_zero_point);
      const __m128i vxa5 =
          sub_zero_point(_mm_unpacklo_epi8(va5, vzero), va_zero_point);
      const __m128i vxa6 =
          sub_zero_point(_mm_unpacklo_epi8(va6, vzero), va_zero_point);
      const __m128i vxa7 =
          sub_zero_point(_mm_unpacklo_epi8(va7, vzero), va_zero_point);

      // acc += a0 * b0;
      __m128i vacc_low_16bits = _mm_mullo_epi16(vxa0, vxb0);
      __m128i vacc_high_16bits = _mm_mulhi_epi16(vxa0, vxb0);
      vacc_low[n] = _mm_add_epi32(vacc_low[n],
        _mm_unpacklo_epi16(vacc_low_16bits, vacc_high_16bits));
      vacc_high[n] = _mm_add_epi32(vacc_high[n],
        _mm_unpackhi_epi16(vacc_low_16bits, vacc_high_16bits));
      // acc += a1 * b1;
      vacc_low_16bits = _mm_mullo_epi16(vxa1, vxb1);
      vacc_high_16bits = _mm_mulhi_epi16(vxa1, vxb1);
      vacc_low[n] = _mm_add_epi32(vacc_low[n],
        _mm_unpacklo_epi16(vacc_low_16bits, vacc_high_16bits));
      vacc_high[n] = _mm_add_epi32(vacc_high[n],
        _mm_unpackhi_epi16(vacc_low_16bits, vacc_high_16bits));
      // acc += a2 * b2;
      vacc_low_16bits = _mm_mullo_epi16(vxa2, vxb2);
      vacc_high_16bits = _mm_mulhi_epi16(vxa2, vxb2);
      vacc_low[n] = _mm_add_epi32(vacc_low[n],
        _mm_unpacklo_epi16(vacc_low_16bits, vacc_high_16bits));
      vacc_high[n] = _mm_add_epi32(vacc_high[n],
        _mm_unpackhi_epi16(vacc_low_16bits, vacc_high_16bits));
      // acc += a3 * b3;
      vacc_low_16bits = _mm_mullo_epi16(vxa3, vxb3);
      vacc_high_16bits = _mm_mulhi_epi16(vxa3, vxb3);
      vacc_low[n] = _mm_add_epi32(vacc_low[n],
        _mm_unpacklo_epi16(vacc_low_16bits, vacc_high_16bits));
      vacc_high[n] = _mm_add_epi32(vacc_high[n],
        _mm_unpackhi_epi16(vacc_low_16bits, vacc_high_16bits));
      // acc += a4 * b4;
      vacc_low_16bits = _mm_mullo_epi16(vxa4, vxb4);
      vacc_high_16bits = _mm_mulhi_epi16(vxa4, vxb4);
      vacc_low[n] = _mm_add_epi32(vacc_low[n],
        _mm_unpacklo_epi16(vacc_low_16bits, vacc_high_16bits));
      vacc_high[n] = _mm_add_epi32(vacc_high[n],
        _mm_unpackhi_epi16(vacc_low_16bits, vacc_high_16bits));
      // acc += a5 * b5;
      vacc_low_16bits = _mm_mullo_epi16(vxa5, vxb5);
      vacc_high_16bits = _mm_mulhi_epi16(vxa5, vxb5);
      vacc_low[n] = _mm_add_epi32(vacc_low[n],
        _mm_unpacklo_epi16(vacc_low_16bits, vacc_high_16bits));
      vacc_high[n] = _mm_add_epi32(vacc_high[n],
        _mm_unpackhi_epi16(vacc_low_16bits, vacc_high_16bits));
      // acc += a6 * b6;
      vacc_low_16bits = _mm_mullo_epi16(vxa6, vxb6);
      vacc_high_16bits = _mm_mulhi_epi16(vxa6, vxb6);
      vacc_low[n] = _mm_add_epi32(vacc_low[n],
        _mm_unpacklo_epi16(vacc_low_16bits, vacc_high_16bits));
      vacc_high[n] = _mm_add_epi32(vacc_high[n],
        _mm_unpackhi_epi16(vacc_low_16bits, vacc_high_16bits));
      // acc += a7 * b7;
      vacc_low_16bits = _mm_mullo_epi16(vxa7, vxb7);
      vacc_high_16bits = _mm_mulhi_epi16(vxa7, vxb7);
      vacc_low[n] = _mm_add_epi32(vacc_low[n],
        _mm_unpacklo_epi16(vacc_low_16bits, vacc_high_16bits));
      vacc_high[n] = _mm_add_epi32(vacc_high[n],
        _mm_unpackhi_epi16(vacc_low_16bits, vacc_high_16bits));

      // Now we have 1x8 m acculated 32 bit values in vacc_low[n](4) and vacc_high[n](4)

      temp_packed_w = temp_packed_w + COL_BLOCK_SIZE * 2;
      temp_w_block_ids_ptr += 2;
      num_blocks -= 2;
    }
    if (num_blocks > 0) {
      // Load two 1x4 uint8 blocks 2 ints
      const uint8_t* b_ptr = temp_packed_w;
      const int16_t b_0 = (int16_t)((uint16_t)(b_ptr[0]));
      const int16_t b_1 = (int16_t)((uint16_t)(b_ptr[1]));
      const int16_t b_2 = (int16_t)((uint16_t)(b_ptr[2]));
      const int16_t b_3 = (int16_t)((uint16_t)(b_ptr[3]));
      // Now we will load 8kx1(broadcast 8) weight values
      const __m128i vxb0 = _mm_set1_epi16((b_0 - b_zero_point));
      const __m128i vxb1 = _mm_set1_epi16((b_1 - b_zero_point));
      const __m128i vxb2 = _mm_set1_epi16((b_2 - b_zero_point));
      const __m128i vxb3 = _mm_set1_epi16((b_3 - b_zero_point));

      // Then load transformed weight blocks
      // 1. Load 4 1x8 registers = 4k x 8m
      // Thus have 4x8 (4k x 8m) activations a0, a1, a2, a3
      // Each a containing 8 m values.

      // Load column id of the first 1x4 block
      int32_t col_block_id_0 = temp_w_block_ids_ptr[0];
      const __m128i va0 =
        _mm_loadl_epi64((const __m128i*) (a_packed +
            col_block_id_0 * PACKED_A_BLOCK_SIZE + MR * 0));
      const __m128i va1 =
        _mm_loadl_epi64((const __m128i*) (a_packed +
            col_block_id_0 * PACKED_A_BLOCK_SIZE + MR * 1));
      const __m128i va2 =
        _mm_loadl_epi64((const __m128i*) (a_packed +
            col_block_id_0 * PACKED_A_BLOCK_SIZE + MR * 2));
      const __m128i va3 =
        _mm_loadl_epi64((const __m128i*) (a_packed +
            col_block_id_0 * PACKED_A_BLOCK_SIZE + MR * 3));
      const __m128i vxa0 =
          sub_zero_point(_mm_unpacklo_epi8(va0, vzero), va_zero_point);
      const __m128i vxa1 =
          sub_zero_point(_mm_unpacklo_epi8(va1, vzero), va_zero_point);
      const __m128i vxa2 =
          sub_zero_point(_mm_unpacklo_epi8(va2, vzero), va_zero_point);
      const __m128i vxa3 =
          sub_zero_point(_mm_unpacklo_epi8(va3, vzero), va_zero_point);

      // acc += a0 * b0;
      __m128i vacc_low_16bits = _mm_mullo_epi16(vxa0, vxb0);
      __m128i vacc_high_16bits = _mm_mulhi_epi16(vxa0, vxb0);
      vacc_low[n] = _mm_add_epi32(vacc_low[n],
        _mm_unpacklo_epi16(vacc_low_16bits, vacc_high_16bits));
      vacc_high[n] = _mm_add_epi32(vacc_high[n],
        _mm_unpackhi_epi16(vacc_low_16bits, vacc_high_16bits));
      // acc += a1 * b1;
      vacc_low_16bits = _mm_mullo_epi16(vxa1, vxb1);
      vacc_high_16bits = _mm_mulhi_epi16(vxa1, vxb1);
      vacc_low[n] = _mm_add_epi32(vacc_low[n],
        _mm_unpacklo_epi16(vacc_low_16bits, vacc_high_16bits));
      vacc_high[n] = _mm_add_epi32(vacc_high[n],
        _mm_unpackhi_epi16(vacc_low_16bits, vacc_high_16bits));
      // acc += a2 * b2;
      vacc_low_16bits = _mm_mullo_epi16(vxa2, vxb2);
      vacc_high_16bits = _mm_mulhi_epi16(vxa2, vxb2);
      vacc_low[n] = _mm_add_epi32(vacc_low[n],
        _mm_unpacklo_epi16(vacc_low_16bits, vacc_high_16bits));
      vacc_high[n] = _mm_add_epi32(vacc_high[n],
        _mm_unpackhi_epi16(vacc_low_16bits, vacc_high_16bits));
      // acc += a3 * b3;
      vacc_low_16bits = _mm_mullo_epi16(vxa3, vxb3);
      vacc_high_16bits = _mm_mulhi_epi16(vxa3, vxb3);
      vacc_low[n] = _mm_add_epi32(vacc_low[n],
        _mm_unpacklo_epi16(vacc_low_16bits, vacc_high_16bits));
      vacc_high[n] = _mm_add_epi32(vacc_high[n],
        _mm_unpackhi_epi16(vacc_low_16bits, vacc_high_16bits));

      // Now we have 1x8 m acculated 32 bit values in vacc_low[n](4) and vacc_high[n](4)
    }
  }

  __m128 vout[8];
  __m128 a_ps, b_ps, c_ps, d_ps, tmp0, tmp1, tmp2, tmp3;

  // Transform low half of 4x8 result
  // That is 4x4 block (4n x 4m)
  // Convert to FP and transpose: 4m x 4n
  CONVERT_TO_FP_AND_TRANSPOSE(vacc_low[0],
                              vacc_low[1],
                              vacc_low[2],
                              vacc_low[3],
                              vout[0],
                              vout[1],
                              vout[2],
                              vout[3])
  CONVERT_TO_FP_AND_TRANSPOSE(vacc_high[0],
                              vacc_high[1],
                              vacc_high[2],
                              vacc_high[3],
                              vout[4],
                              vout[5],
                              vout[6],
                              vout[7])

  vout[0] = _mm_mul_ps(vmultiplier, vout[0]);
  vout[1] = _mm_mul_ps(vmultiplier, vout[1]);
  vout[2] = _mm_mul_ps(vmultiplier, vout[2]);
  vout[3] = _mm_mul_ps(vmultiplier, vout[3]);
  vout[4] = _mm_mul_ps(vmultiplier, vout[4]);
  vout[5] = _mm_mul_ps(vmultiplier, vout[5]);
  vout[6] = _mm_mul_ps(vmultiplier, vout[6]);
  vout[7] = _mm_mul_ps(vmultiplier, vout[7]);

  vout[0] = _mm_add_ps(vout[0], vbias);
  vout[1] = _mm_add_ps(vout[1], vbias);
  vout[2] = _mm_add_ps(vout[2], vbias);
  vout[3] = _mm_add_ps(vout[3], vbias);
  vout[4] = _mm_add_ps(vout[4], vbias);
  vout[5] = _mm_add_ps(vout[5], vbias);
  vout[6] = _mm_add_ps(vout[6], vbias);
  vout[7] = _mm_add_ps(vout[7], vbias);

  float* c0 = c;
  float* c1 = c0 + c_stride;
  if (mr < 2) {
    c1 = c0;
    vout[1] = vout[0];
  }
  float* c2 = c1 + c_stride;
  if (mr < 3) {
    c2 = c0;
    vout[2] = vout[0];
  }
  float* c3 = c2 + c_stride;
  if (mr < 4) {
    c3 = c0;
    vout[3] = vout[0];
  }
  float* c4 = c3 + c_stride;
  if (mr < 5) {
    c4 = c0;
    vout[4] = vout[0];
  }
  float* c5 = c4 + c_stride;
  if (mr < 6) {
    c5 = c0;
    vout[5] = vout[0];
  }
  float* c6 = c5 + c_stride;
  if (mr < 7) {
    c6 = c0;
    vout[6] = vout[0];
  }
  float* c7 = c6 + c_stride;
  if (mr < 8) {
    c7 = c0;
    vout[7] = vout[0];
  }

  if (nr == 4) {
    _mm_storeu_ps(c0, vout[0]);
    _mm_storeu_ps(c1, vout[1]);
    _mm_storeu_ps(c2, vout[2]);
    _mm_storeu_ps(c3, vout[3]);
    _mm_storeu_ps(c4, vout[4]);
    _mm_storeu_ps(c5, vout[5]);
    _mm_storeu_ps(c6, vout[6]);
    _mm_storeu_ps(c7, vout[7]);
  } else {
    if (nr >= 2) {
      _mm_storel_pi((__m64*)c0, vout[0]);
      _mm_storel_pi((__m64*)c1, vout[1]);
      _mm_storel_pi((__m64*)c2, vout[2]);
      _mm_storel_pi((__m64*)c3, vout[3]);
      _mm_storel_pi((__m64*)c4, vout[4]);
      _mm_storel_pi((__m64*)c5, vout[5]);
      _mm_storel_pi((__m64*)c6, vout[6]);
      _mm_storel_pi((__m64*)c7, vout[7]);

      nr -= 2;

      c0 += 2;
      c1 += 2;
      c2 += 2;
      c3 += 2;
      c4 += 2;
      c5 += 2;
      c6 += 2;
      c7 += 2;
      vout[0] = _mm_shuffle_ps(vout[0], vout[0], _MM_SHUFFLE(2, 2, 2, 2));
      vout[1] = _mm_shuffle_ps(vout[1], vout[1], _MM_SHUFFLE(2, 2, 2, 2));
      vout[2] = _mm_shuffle_ps(vout[2], vout[2], _MM_SHUFFLE(2, 2, 2, 2));
      vout[3] = _mm_shuffle_ps(vout[3], vout[3], _MM_SHUFFLE(2, 2, 2, 2));
      vout[4] = _mm_shuffle_ps(vout[4], vout[4], _MM_SHUFFLE(2, 2, 2, 2));
      vout[5] = _mm_shuffle_ps(vout[5], vout[5], _MM_SHUFFLE(2, 2, 2, 2));
      vout[6] = _mm_shuffle_ps(vout[6], vout[6], _MM_SHUFFLE(2, 2, 2, 2));
      vout[7] = _mm_shuffle_ps(vout[7], vout[7], _MM_SHUFFLE(2, 2, 2, 2));
    }
    if (nr != 0) {
      *c0 = _mm_cvtss_f32(vout[0]);
      *c1 = _mm_cvtss_f32(vout[1]);
      *c2 = _mm_cvtss_f32(vout[2]);
      *c3 = _mm_cvtss_f32(vout[3]);
      *c4 = _mm_cvtss_f32(vout[4]);
      *c5 = _mm_cvtss_f32(vout[5]);
      *c6 = _mm_cvtss_f32(vout[6]);
      *c7 = _mm_cvtss_f32(vout[7]);
    }
  }
}
