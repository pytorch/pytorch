#pragma once

#ifdef __aarch64__

#include <arm_neon.h>
#include <cstdint>

namespace at::vec {
inline namespace CPU_CAPABILITY {

static inline void transpose_kernel_8x8_neon(
    const float* src,
    int64_t ld_src,
    float* dst,
    int64_t ld_dst) {
  float32x4_t q0 = vld1q_f32(src);
  float32x4_t q1 = vld1q_f32(src + 4);
  src += ld_src;
  float32x4_t q2 = vld1q_f32(src);
  float32x4_t q3 = vld1q_f32(src + 4);
  src += ld_src;
  float32x4_t q4 = vld1q_f32(src);
  float32x4_t q5 = vld1q_f32(src + 4);
  src += ld_src;
  float32x4_t q6 = vld1q_f32(src);
  float32x4_t q7 = vld1q_f32(src + 4);
  src += ld_src;
  float32x4_t q16 = vld1q_f32(src);
  float32x4_t q17 = vld1q_f32(src + 4);
  src += ld_src;
  float32x4_t q18 = vld1q_f32(src);
  float32x4_t q19 = vld1q_f32(src + 4);
  src += ld_src;
  float32x4_t q20 = vld1q_f32(src);
  float32x4_t q21 = vld1q_f32(src + 4);
  src += ld_src;
  float32x4_t q22 = vld1q_f32(src);
  float32x4_t q23 = vld1q_f32(src + 4);

  float32x4_t v24 = vzip1q_f32(q0, q2);
  float32x4_t v25 = vzip1q_f32(q4, q6);
  float32x4_t v26 = vzip1q_f32(q16, q18);
  float32x4_t v27 = vzip1q_f32(q20, q22);

  float64x2_t v28 = vzip1q_f64(vreinterpretq_f64_f32(v24), vreinterpretq_f64_f32(v25));
  float64x2_t v29 = vzip1q_f64(vreinterpretq_f64_f32(v26), vreinterpretq_f64_f32(v27));

  vst1q_f32(dst, vreinterpretq_f32_f64(v28));
  vst1q_f32(dst + 4, vreinterpretq_f32_f64(v29));
  dst += ld_dst;

  float64x2_t v30 = vzip2q_f64(vreinterpretq_f64_f32(v24), vreinterpretq_f64_f32(v25));
  float64x2_t v31 = vzip2q_f64(vreinterpretq_f64_f32(v26), vreinterpretq_f64_f32(v27));

  vst1q_f32(dst, vreinterpretq_f32_f64(v30));
  vst1q_f32(dst + 4, vreinterpretq_f32_f64(v31));
  dst += ld_dst;

  v24 = vzip2q_f32(q0, q2);
  v25 = vzip2q_f32(q4, q6);
  v26 = vzip2q_f32(q16, q18);
  v27 = vzip2q_f32(q20, q22);

  v28 = vzip1q_f64(vreinterpretq_f64_f32(v24), vreinterpretq_f64_f32(v25));
  v29 = vzip1q_f64(vreinterpretq_f64_f32(v26), vreinterpretq_f64_f32(v27));

  vst1q_f32(dst, vreinterpretq_f32_f64(v28));
  vst1q_f32(dst + 4, vreinterpretq_f32_f64(v29));
  dst += ld_dst;

  v30 = vzip2q_f64(vreinterpretq_f64_f32(v24), vreinterpretq_f64_f32(v25));
  v31 = vzip2q_f64(vreinterpretq_f64_f32(v26), vreinterpretq_f64_f32(v27));

  vst1q_f32(dst, vreinterpretq_f32_f64(v30));
  vst1q_f32(dst + 4, vreinterpretq_f32_f64(v31));
  dst += ld_dst;

  v24 = vzip1q_f32(q1, q3);
  v25 = vzip1q_f32(q5, q7);
  v26 = vzip1q_f32(q17, q19);
  v27 = vzip1q_f32(q21, q23);

  v28 = vzip1q_f64(vreinterpretq_f64_f32(v24), vreinterpretq_f64_f32(v25));
  v29 = vzip1q_f64(vreinterpretq_f64_f32(v26), vreinterpretq_f64_f32(v27));

  vst1q_f32(dst, vreinterpretq_f32_f64(v28));
  vst1q_f32(dst + 4, vreinterpretq_f32_f64(v29));
  dst += ld_dst;

  v30 = vzip2q_f64(vreinterpretq_f64_f32(v24), vreinterpretq_f64_f32(v25));
  v31 = vzip2q_f64(vreinterpretq_f64_f32(v26), vreinterpretq_f64_f32(v27));

  vst1q_f32(dst, vreinterpretq_f32_f64(v30));
  vst1q_f32(dst + 4, vreinterpretq_f32_f64(v31));
  dst += ld_dst;

  v24 = vzip2q_f32(q1, q3);
  v25 = vzip2q_f32(q5, q7);
  v26 = vzip2q_f32(q17, q19);
  v27 = vzip2q_f32(q21, q23);

  v28 = vzip1q_f64(vreinterpretq_f64_f32(v24), vreinterpretq_f64_f32(v25));
  v29 = vzip1q_f64(vreinterpretq_f64_f32(v26), vreinterpretq_f64_f32(v27));

  vst1q_f32(dst, vreinterpretq_f32_f64(v28));
  vst1q_f32(dst + 4, vreinterpretq_f32_f64(v29));
  dst += ld_dst;

  v30 = vzip2q_f64(vreinterpretq_f64_f32(v24), vreinterpretq_f64_f32(v25));
  v31 = vzip2q_f64(vreinterpretq_f64_f32(v26), vreinterpretq_f64_f32(v27));

  vst1q_f32(dst, vreinterpretq_f32_f64(v30));
  vst1q_f32(dst + 4, vreinterpretq_f32_f64(v31));
}

static inline void transpose_kernel_8x4_neon(
    const float* src,
    int64_t ld_src,
    float* dst,
    int64_t ld_dst) {
  float32x4_t q0 = vld1q_f32(src);
  src += ld_src;
  float32x4_t q1 = vld1q_f32(src);
  src += ld_src;
  float32x4_t q2 = vld1q_f32(src);
  src += ld_src;
  float32x4_t q3 = vld1q_f32(src);
  src += ld_src;
  float32x4_t q4 = vld1q_f32(src);
  src += ld_src;
  float32x4_t q5 = vld1q_f32(src);
  src += ld_src;
  float32x4_t q6 = vld1q_f32(src);
  src += ld_src;
  float32x4_t q7 = vld1q_f32(src);

  float32x4_t v16 = vzip1q_f32(q0, q1);
  float32x4_t v17 = vzip1q_f32(q2, q3);
  float32x4_t v18 = vzip1q_f32(q4, q5);
  float32x4_t v19 = vzip1q_f32(q6, q7);

  float64x2_t v20 = vzip1q_f64(vreinterpretq_f64_f32(v16), vreinterpretq_f64_f32(v17));
  float64x2_t v21 = vzip1q_f64(vreinterpretq_f64_f32(v18), vreinterpretq_f64_f32(v19));

  vst1q_f32(dst, vreinterpretq_f32_f64(v20));
  vst1q_f32(dst + 4, vreinterpretq_f32_f64(v21));
  dst += ld_dst;

  float64x2_t v22 = vzip2q_f64(vreinterpretq_f64_f32(v16), vreinterpretq_f64_f32(v17));
  float64x2_t v23 = vzip2q_f64(vreinterpretq_f64_f32(v18), vreinterpretq_f64_f32(v19));

  vst1q_f32(dst, vreinterpretq_f32_f64(v22));
  vst1q_f32(dst + 4, vreinterpretq_f32_f64(v23));
  dst += ld_dst;

  float32x4_t v24 = vzip2q_f32(q0, q1);
  float32x4_t v25 = vzip2q_f32(q2, q3);
  float32x4_t v26 = vzip2q_f32(q4, q5);
  float32x4_t v27 = vzip2q_f32(q6, q7);

  float64x2_t v28 = vzip1q_f64(vreinterpretq_f64_f32(v24), vreinterpretq_f64_f32(v25));
  float64x2_t v29 = vzip1q_f64(vreinterpretq_f64_f32(v26), vreinterpretq_f64_f32(v27));

  vst1q_f32(dst, vreinterpretq_f32_f64(v28));
  vst1q_f32(dst + 4, vreinterpretq_f32_f64(v29));
  dst += ld_dst;

  float64x2_t v30 = vzip2q_f64(vreinterpretq_f64_f32(v24), vreinterpretq_f64_f32(v25));
  float64x2_t v31 = vzip2q_f64(vreinterpretq_f64_f32(v26), vreinterpretq_f64_f32(v27));

  vst1q_f32(dst, vreinterpretq_f32_f64(v30));
  vst1q_f32(dst + 4, vreinterpretq_f32_f64(v31));
}

static inline void transpose_kernel_8x2_neon(
    const float* src,
    int64_t ld_src,
    float* dst,
    int64_t ld_dst) {
  float32x2_t q0 = vld1_f32(src);
  src += ld_src;
  float32x2_t q1 = vld1_f32(src);
  src += ld_src;
  float32x2_t q2 = vld1_f32(src);
  src += ld_src;
  float32x2_t q3 = vld1_f32(src);
  src += ld_src;
  float32x2_t q4 = vld1_f32(src);
  src += ld_src;
  float32x2_t q5 = vld1_f32(src);
  src += ld_src;
  float32x2_t q6 = vld1_f32(src);
  src += ld_src;
  float32x2_t q7 = vld1_f32(src);

  float32x2_t v16 = vzip1_f32(q0, q1);
  float32x2_t v17 = vzip1_f32(q2, q3);
  float32x2_t v18 = vzip1_f32(q4, q5);
  float32x2_t v19 = vzip1_f32(q6, q7);

  float32x4_t q16;
  q16[0] = v16[0];
  q16[1] = v16[1];
  float32x4_t q17;
  q17[0] = v17[0];
  q17[1] = v17[1];
  float32x4_t q18;
  q18[0] = v18[0];
  q18[1] = v18[1];
  float32x4_t q19;
  q19[0] = v19[0];
  q19[1] = v19[1];

  float64x2_t v20 = vzip1q_f64(vreinterpretq_f64_f32(q16), vreinterpretq_f64_f32(q17));
  float64x2_t v21 = vzip1q_f64(vreinterpretq_f64_f32(q18), vreinterpretq_f64_f32(q19));

  vst1q_f32(dst, vreinterpretq_f32_f64(v20));
  vst1q_f32(dst + 4, vreinterpretq_f32_f64(v21));
  dst += ld_dst;

  float32x2_t v22 = vzip2_f32(q0, q1);
  float32x2_t v23 = vzip2_f32(q2, q3);
  float32x2_t v24 = vzip2_f32(q4, q5);
  float32x2_t v25 = vzip2_f32(q6, q7);

  float32x4_t q22;
  q22[0] = v22[0];
  q22[1] = v22[1];
  float32x4_t q23;
  q23[0] = v23[0];
  q23[1] = v23[1];
  float32x4_t q24;
  q24[0] = v24[0];
  q24[1] = v24[1];
  float32x4_t q25;
  q25[0] = v25[0];
  q25[1] = v25[1];

  float64x2_t v26 = vzip1q_f64(vreinterpretq_f64_f32(q22), vreinterpretq_f64_f32(q23));
  float64x2_t v27 = vzip1q_f64(vreinterpretq_f64_f32(q24), vreinterpretq_f64_f32(q25));

  vst1q_f32(dst, vreinterpretq_f32_f64(v26));
  vst1q_f32(dst + 4, vreinterpretq_f32_f64(v27));
}

static inline void transpose_kernel_4x8_neon(
    const float* src,
    int64_t ld_src,
    float* dst,
    int64_t ld_dst) {
  float32x4_t q0 = vld1q_f32(src);
  float32x4_t q1 = vld1q_f32(src + 4);
  src += ld_src;
  float32x4_t q2 = vld1q_f32(src);
  float32x4_t q3 = vld1q_f32(src + 4);
  src += ld_src;
  float32x4_t q4 = vld1q_f32(src);
  float32x4_t q5 = vld1q_f32(src + 4);
  src += ld_src;
  float32x4_t q6 = vld1q_f32(src);
  float32x4_t q7 = vld1q_f32(src + 4);

  float32x4_t v16 = vzip1q_f32(q0, q2);
  float32x4_t v17 = vzip1q_f32(q4, q6);

  float64x2_t v18 = vzip1q_f64(vreinterpretq_f64_f32(v16), vreinterpretq_f64_f32(v17));

  vst1q_f32(dst, vreinterpretq_f32_f64(v18));
  dst += ld_dst;

  float64x2_t v19 = vzip2q_f64(vreinterpretq_f64_f32(v16), vreinterpretq_f64_f32(v17));

  vst1q_f32(dst, vreinterpretq_f32_f64(v19));
  dst += ld_dst;

  float32x4_t v20 = vzip2q_f32(q0, q2);
  float32x4_t v21 = vzip2q_f32(q4, q6);

  float64x2_t v22 = vzip1q_f64(vreinterpretq_f64_f32(v20), vreinterpretq_f64_f32(v21));

  vst1q_f32(dst, vreinterpretq_f32_f64(v22));
  dst += ld_dst;

  float64x2_t v23 = vzip2q_f64(vreinterpretq_f64_f32(v20), vreinterpretq_f64_f32(v21));

  vst1q_f32(dst, vreinterpretq_f32_f64(v23));
  dst += ld_dst;

  float32x4_t v24 = vzip1q_f32(q1, q3);
  float32x4_t v25 = vzip1q_f32(q5, q7);

  float64x2_t v26 = vzip1q_f64(vreinterpretq_f64_f32(v24), vreinterpretq_f64_f32(v25));

  vst1q_f32(dst, vreinterpretq_f32_f64(v26));
  dst += ld_dst;

  float64x2_t v27 = vzip2q_f64(vreinterpretq_f64_f32(v24), vreinterpretq_f64_f32(v25));

  vst1q_f32(dst, vreinterpretq_f32_f64(v27));
  dst += ld_dst;

  float32x4_t v28 = vzip2q_f32(q1, q3);
  float32x4_t v29 = vzip2q_f32(q5, q7);

  float64x2_t v30 = vzip1q_f64(vreinterpretq_f64_f32(v28), vreinterpretq_f64_f32(v29));

  vst1q_f32(dst, vreinterpretq_f32_f64(v30));
  dst += ld_dst;

  float64x2_t v31 = vzip2q_f64(vreinterpretq_f64_f32(v28), vreinterpretq_f64_f32(v29));

  vst1q_f32(dst, vreinterpretq_f32_f64(v31));
}

static inline void transpose_kernel_4x4_neon(
    const float* src,
    int64_t ld_src,
    float* dst,
    int64_t ld_dst) {
  float32x4_t q0 = vld1q_f32(src);
  src += ld_src;
  float32x4_t q1 = vld1q_f32(src);
  src += ld_src;
  float32x4_t q2 = vld1q_f32(src);
  src += ld_src;
  float32x4_t q3 = vld1q_f32(src);

  float32x4_t v4 = vzip1q_f32(q0, q1);
  float32x4_t v5 = vzip1q_f32(q2, q3);

  float64x2_t v6 = vzip1q_f64(vreinterpretq_f64_f32(v4), vreinterpretq_f64_f32(v5));

  vst1q_f32(dst, vreinterpretq_f32_f64(v6));
  dst += ld_dst;

  float64x2_t v7 = vzip2q_f64(vreinterpretq_f64_f32(v4), vreinterpretq_f64_f32(v5));

  vst1q_f32(dst, vreinterpretq_f32_f64(v7));
  dst += ld_dst;

  float32x4_t v8 = vzip2q_f32(q0, q1);
  float32x4_t v9 = vzip2q_f32(q2, q3);

  float64x2_t v10 = vzip1q_f64(vreinterpretq_f64_f32(v8), vreinterpretq_f64_f32(v9));

  vst1q_f32(dst, vreinterpretq_f32_f64(v10));
  dst += ld_dst;

  float64x2_t v11 = vzip2q_f64(vreinterpretq_f64_f32(v8), vreinterpretq_f64_f32(v9));

  vst1q_f32(dst, vreinterpretq_f32_f64(v11));
}

static inline void transpose_kernel_4x2_neon(
    const float* src,
    int64_t ld_src,
    float* dst,
    int64_t ld_dst) {
  float32x2_t d0 = vld1_f32(src);
  src += ld_src;
  float32x2_t d1 = vld1_f32(src);
  src += ld_src;
  float32x2_t d2 = vld1_f32(src);
  src += ld_src;
  float32x2_t d3 = vld1_f32(src);

  float32x2_t v4 = vzip1_f32(d0, d1);
  float32x2_t v5 = vzip1_f32(d2, d3);

  float32x4_t q4;
  q4[0] = v4[0];
  q4[1] = v4[1];
  float32x4_t q5;
  q5[0] = v5[0];
  q5[1] = v5[1];

  float64x2_t v6 = vzip1q_f64(vreinterpretq_f64_f32(q4), vreinterpretq_f64_f32(q5));

  vst1q_f64(reinterpret_cast<double*>(dst), v6);
  dst += ld_dst;

  float32x2_t v7 = vzip2_f32(d0, d1);
  float32x2_t v8 = vzip2_f32(d2, d3);

  float32x4_t q7;
  q7[0] = v7[0];
  q7[1] = v7[1];
  float32x4_t q8;
  q8[0] = v8[0];
  q8[1] = v8[1];

  float64x2_t v9 = vzip1q_f64(vreinterpretq_f64_f32(q7), vreinterpretq_f64_f32(q8));

  vst1q_f64(reinterpret_cast<double*>(dst), v9);
}

static inline void transpose_kernel_2x8_neon(
    const float* src,
    int64_t ld_src,
    float* dst,
    int64_t ld_dst) {
  float32x4_t q0 = vld1q_f32(src);
  float32x4_t q1 = vld1q_f32(src + 4);
  src += ld_src;
  float32x4_t q2 = vld1q_f32(src);
  float32x4_t q3 = vld1q_f32(src + 4);

  float32x4_t v4 = vzip1q_f32(q0, q2);

  vst1q_lane_f64(reinterpret_cast<double*>(dst), vreinterpretq_f64_f32(v4), 0);
  dst += ld_dst;

  vst1q_lane_f64(reinterpret_cast<double*>(dst), vreinterpretq_f64_f32(v4), 1);
  dst += ld_dst;

  float32x4_t v5 = vzip2q_f32(q0, q2);

  vst1q_lane_f64(reinterpret_cast<double*>(dst), vreinterpretq_f64_f32(v5), 0);
  dst += ld_dst;

  vst1q_lane_f64(reinterpret_cast<double*>(dst), vreinterpretq_f64_f32(v5), 1);
  dst += ld_dst;

  float32x4_t v6 = vzip1q_f32(q1, q3);

  vst1q_lane_f64(reinterpret_cast<double*>(dst), vreinterpretq_f64_f32(v6), 0);
  dst += ld_dst;

  vst1q_lane_f64(reinterpret_cast<double*>(dst), vreinterpretq_f64_f32(v6), 1);
  dst += ld_dst;

  float32x4_t v7 = vzip2q_f32(q1, q3);

  vst1q_lane_f64(reinterpret_cast<double*>(dst), vreinterpretq_f64_f32(v7), 0);
  dst += ld_dst;

  vst1q_lane_f64(reinterpret_cast<double*>(dst), vreinterpretq_f64_f32(v7), 1);
}

static inline void transpose_kernel_2x4_neon(
    const float* src,
    int64_t ld_src,
    float* dst,
    int64_t ld_dst) {

  float32x4_t q0 = vld1q_f32(src);
  src += ld_src;
  float32x4_t q1 = vld1q_f32(src);

  float32x4_t v3 = vzip1q_f32(q0, q1);

  vst1q_lane_f64(reinterpret_cast<double*>(dst), vreinterpretq_f64_f32(v3), 0);
  dst += ld_dst;

  vst1q_lane_f64(reinterpret_cast<double*>(dst), vreinterpretq_f64_f32(v3), 1);
  dst += ld_dst;

  float32x4_t v4 = vzip2q_f32(q0, q1);

  vst1q_lane_f64(reinterpret_cast<double*>(dst), vreinterpretq_f64_f32(v4), 0);
  dst += ld_dst;

  vst1q_lane_f64(reinterpret_cast<double*>(dst), vreinterpretq_f64_f32(v4), 1);
}

static inline void transpose_kernel_2x2_neon(
    const float* src,
    int64_t ld_src,
    float* dst,
    int64_t ld_dst) {
  float32x2_t d0 = vld1_f32(src);
  src += ld_src;
  float32x2_t d1 = vld1_f32(src);

  float32x4_t q0;
  q0[0] = d0[0];
  q0[1] = d0[1];

  float32x4_t q1;
  q1[0] = d1[0];
  q1[1] = d1[1];

  float32x4_t v3 = vzip1q_f32(q0, q1);

  vst1q_lane_f64(reinterpret_cast<double*>(dst), vreinterpretq_f64_f32(v3), 0);
  dst += ld_dst;

  vst1q_lane_f64(reinterpret_cast<double*>(dst), vreinterpretq_f64_f32(v3), 1);
}

static inline void transpose_kernel_mx1(
    const float* src,
    int64_t ld_src,
    float* dst,
    const int64_t M) {
  for (int64_t i = 0; i < M; ++i) {
    dst[i] = src[i * ld_src];
  }
}

inline void transpose_float_neon(
    int64_t M,
    int64_t N,
    const float* src,
    int64_t ld_src,
    float* dst,
    int64_t ld_dst) {
  int64_t jb = 0;
  while (jb + 7 < M) {
    int64_t ib = 0;
    while (ib + 7 < N) {
      transpose_kernel_8x8_neon(
          &src[ib + jb * ld_src], ld_src, &dst[jb + ib * ld_dst], ld_dst);
      ib += 8;
    }
    while (ib + 3 < N) {
      transpose_kernel_8x4_neon(
          &src[ib + jb * ld_src], ld_src, &dst[jb + ib * ld_dst], ld_dst);
      ib += 4;
    }
    while (ib + 1 < N) {
      transpose_kernel_8x2_neon(
          &src[ib + jb * ld_src], ld_src, &dst[jb + ib * ld_dst], ld_dst);
      ib += 2;
    }
    if (ib < N) {
      transpose_kernel_mx1(
          &src[ib + jb * ld_src], ld_src, &dst[jb + ib * ld_dst], 8);
    }
    jb += 8;
  }
  while (jb + 3 < M) {
    int64_t ib = 0;
    while (ib + 7 < N) {
      transpose_kernel_4x8_neon(
          &src[ib + jb * ld_src], ld_src, &dst[jb + ib * ld_dst], ld_dst);
      ib += 8;
    }
    while (ib + 3 < N) {
      transpose_kernel_4x4_neon(
          &src[ib + jb * ld_src], ld_src, &dst[jb + ib * ld_dst], ld_dst);
      ib += 4;
    }
    while (ib + 1 < N) {
      transpose_kernel_4x2_neon(
          &src[ib + jb * ld_src], ld_src, &dst[jb + ib * ld_dst], ld_dst);
      ib += 2;
    }
    if (ib < N) {
      transpose_kernel_mx1(
          &src[ib + jb * ld_src], ld_src, &dst[jb + ib * ld_dst], 4);
    }
    jb += 4;
  }
  while (jb + 1 < M) {
    int64_t ib = 0;
    while (ib + 7 < N) {
      transpose_kernel_2x8_neon(
          &src[ib + jb * ld_src], ld_src, &dst[jb + ib * ld_dst], ld_dst);
      ib += 8;
    }
    while (ib + 3 < N) {
      transpose_kernel_2x4_neon(
          &src[ib + jb * ld_src], ld_src, &dst[jb + ib * ld_dst], ld_dst);
      ib += 4;
    }
    while (ib + 1 < N) {
      transpose_kernel_2x2_neon(
          &src[ib + jb * ld_src], ld_src, &dst[jb + ib * ld_dst], ld_dst);
      ib += 2;
    }
    if (ib < N) {
      transpose_kernel_mx1(
          &src[ib + jb * ld_src], ld_src, &dst[jb + ib * ld_dst], 2);
    }
    jb += 2;
  }
  if (jb < M) {
    for (int64_t ib = 0; ib < N; ++ib) {
      dst[jb + ib * ld_dst] = src[ib + jb * ld_src];
    }
  }
}

} // namespace CPU_CAPABILITY
} // namespace at::vec

#endif /* __aarch64__ */
