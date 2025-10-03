#pragma once

#include <arm_neon.h>
#include <cstdint>

namespace at::vec {
inline namespace CPU_CAPABILITY {

static inline void transpose_kernel_8x8_neon(
    const float* src,
    int64_t ld_src,
    float* dst,
    int64_t ld_dst) {
  asm volatile(
      "mov x0, %[src]\t\n"
      "mov x1, %[ld_src]\t\n"
      "mov x2, %[dst]\t\n"
      "mov x3, %[ld_dst]\t\n"

      "ldp q0, q1, [x0]\t\n"
      "lsl x1, x1, #2\t\n"
      "add x0, x0, x1\t\n"
      "ldp q2, q3, [x0]\t\n"
      "add x0, x0, x1\t\n"
      "ldp q4, q5, [x0]\t\n"
      "add x0, x0, x1\t\n"
      "ldp q6, q7, [x0]\t\n"
      "add x0, x0, x1\t\n"
      "ldp q16, q17, [x0]\t\n"
      "add x0, x0, x1\t\n"
      "ldp q18, q19, [x0]\t\n"
      "add x0, x0, x1\t\n"
      "ldp q20, q21, [x0]\t\n"
      "add x0, x0, x1\t\n"
      "ldp q22, q23, [x0]\t\n"

      "zip1 v24.4s, v0.4s, v2.4s\t\n"
      "zip1 v25.4s, v4.4s, v6.4s\t\n"
      "zip1 v26.4s, v16.4s, v18.4s\t\n"
      "zip1 v27.4s, v20.4s, v22.4s\t\n"

      "zip1 v28.2d, v24.2d, v25.2d\t\n"
      "zip1 v29.2d, v26.2d, v27.2d\t\n"
      "lsl x3, x3, #2\t\n"
      "stp q28, q29, [x2]\t\n"

      "zip2 v30.2d, v24.2d, v25.2d\t\n"
      "zip2 v31.2d, v26.2d, v27.2d\t\n"
      "add x2, x2, x3\t\n"
      "stp q30, q31, [x2]\t\n"

      "zip2 v24.4s, v0.4s, v2.4s\t\n"
      "zip2 v25.4s, v4.4s, v6.4s\t\n"
      "zip2 v26.4s, v16.4s, v18.4s\t\n"
      "zip2 v27.4s, v20.4s, v22.4s\t\n"

      "zip1 v28.2d, v24.2d, v25.2d\t\n"
      "zip1 v29.2d, v26.2d, v27.2d\t\n"
      "add x2, x2, x3\t\n"
      "stp q28, q29, [x2]\t\n"

      "zip2 v30.2d, v24.2d, v25.2d\t\n"
      "zip2 v31.2d, v26.2d, v27.2d\t\n"
      "add x2, x2, x3\t\n"
      "stp q30, q31, [x2]\t\n"

      "zip1 v24.4s, v1.4s, v3.4s\t\n"
      "zip1 v25.4s, v5.4s, v7.4s\t\n"
      "zip1 v26.4s, v17.4s, v19.4s\t\n"
      "zip1 v27.4s, v21.4s, v23.4s\t\n"

      "zip1 v28.2d, v24.2d, v25.2d\t\n"
      "zip1 v29.2d, v26.2d, v27.2d\t\n"
      "add x2, x2, x3\t\n"
      "stp q28, q29, [x2]\t\n"

      "zip2 v30.2d, v24.2d, v25.2d\t\n"
      "zip2 v31.2d, v26.2d, v27.2d\t\n"
      "add x2, x2, x3\t\n"
      "stp q30, q31, [x2]\t\n"

      "zip2 v24.4s, v1.4s, v3.4s\t\n"
      "zip2 v25.4s, v5.4s, v7.4s\t\n"
      "zip2 v26.4s, v17.4s, v19.4s\t\n"
      "zip2 v27.4s, v21.4s, v23.4s\t\n"

      "zip1 v28.2d, v24.2d, v25.2d\t\n"
      "zip1 v29.2d, v26.2d, v27.2d\t\n"
      "add x2, x2, x3\t\n"
      "stp q28, q29, [x2]\t\n"

      "zip2 v30.2d, v24.2d, v25.2d\t\n"
      "zip2 v31.2d, v26.2d, v27.2d\t\n"
      "add x2, x2, x3\t\n"
      "stp q30, q31, [x2]\t\n"

      :
      :
      [src] "r"(src), [ld_src] "r"(ld_src), [dst] "r"(dst), [ld_dst] "r"(ld_dst)
      : "memory",
        "cc",
        "x0",
        "x1",
        "x2",
        "x3",
        "v0",
        "v1",
        "v2",
        "v3",
        "v4",
        "v5",
        "v6",
        "v7",
        "v16",
        "v17",
        "v18",
        "v19",
        "v20",
        "v21",
        "v22",
        "v23",
        "v24",
        "v25",
        "v26",
        "v27",
        "v28",
        "v29",
        "v30",
        "v31");
}

static inline void transpose_kernel_8x4_neon(
    const float* src,
    int64_t ld_src,
    float* dst,
    int64_t ld_dst) {
  asm volatile(
      "mov x0, %[src]\t\n"
      "mov x1, %[ld_src]\t\n"
      "mov x2, %[dst]\t\n"
      "mov x3, %[ld_dst]\t\n"

      "ldr q0, [x0]\t\n"
      "lsl x1, x1, #2\t\n"
      "add x0, x0, x1\t\n"
      "ldr q1, [x0]\t\n"
      "add x0, x0, x1\t\n"
      "ldr q2, [x0]\t\n"
      "add x0, x0, x1\t\n"
      "ldr q3, [x0]\t\n"
      "add x0, x0, x1\t\n"
      "ldr q4, [x0]\t\n"
      "add x0, x0, x1\t\n"
      "ldr q5, [x0]\t\n"
      "add x0, x0, x1\t\n"
      "ldr q6, [x0]\t\n"
      "add x0, x0, x1\t\n"
      "ldr q7, [x0]\t\n"

      "zip1 v16.4s, v0.4s, v1.4s\t\n"
      "zip1 v17.4s, v2.4s, v3.4s\t\n"
      "zip1 v18.4s, v4.4s, v5.4s\t\n"
      "zip1 v19.4s, v6.4s, v7.4s\t\n"

      "zip1 v20.2d, v16.2d, v17.2d\t\n"
      "zip1 v21.2d, v18.2d, v19.2d\t\n"
      "lsl x3, x3, #2\t\n"
      "stp q20, q21, [x2]\t\n"

      "zip2 v22.2d, v16.2d, v17.2d\t\n"
      "zip2 v23.2d, v18.2d, v19.2d\t\n"
      "add x2, x2, x3\t\n"
      "stp q22, q23, [x2]\t\n"

      "zip2 v24.4s, v0.4s, v1.4s\t\n"
      "zip2 v25.4s, v2.4s, v3.4s\t\n"
      "zip2 v26.4s, v4.4s, v5.4s\t\n"
      "zip2 v27.4s, v6.4s, v7.4s\t\n"

      "zip1 v28.2d, v24.2d, v25.2d\t\n"
      "zip1 v29.2d, v26.2d, v27.2d\t\n"
      "add x2, x2, x3\t\n"
      "stp q28, q29, [x2]\t\n"

      "zip2 v30.2d, v24.2d, v25.2d\t\n"
      "zip2 v31.2d, v26.2d, v27.2d\t\n"
      "add x2, x2, x3\t\n"
      "stp q30, q31, [x2]\t\n"

      :
      :
      [src] "r"(src), [ld_src] "r"(ld_src), [dst] "r"(dst), [ld_dst] "r"(ld_dst)
      : "memory",
        "cc",
        "x0",
        "x1",
        "x2",
        "x3",
        "v0",
        "v1",
        "v2",
        "v3",
        "v4",
        "v5",
        "v6",
        "v7",
        "v16",
        "v17",
        "v18",
        "v19",
        "v20",
        "v21",
        "v22",
        "v23",
        "v24",
        "v25",
        "v26",
        "v27",
        "v28",
        "v29",
        "v30",
        "v31");
}

static inline void transpose_kernel_8x2_neon(
    const float* src,
    int64_t ld_src,
    float* dst,
    int64_t ld_dst) {
  asm volatile(
      "mov x0, %[src]\t\n"
      "mov x1, %[ld_src]\t\n"
      "mov x2, %[dst]\t\n"
      "mov x3, %[ld_dst]\t\n"

      "ldr d0, [x0]\t\n"
      "lsl x1, x1, #2\t\n"
      "add x0, x0, x1\t\n"
      "ldr d1, [x0]\t\n"
      "add x0, x0, x1\t\n"
      "ldr d2, [x0]\t\n"
      "add x0, x0, x1\t\n"
      "ldr d3, [x0]\t\n"
      "add x0, x0, x1\t\n"
      "ldr d4, [x0]\t\n"
      "add x0, x0, x1\t\n"
      "ldr d5, [x0]\t\n"
      "add x0, x0, x1\t\n"
      "ldr d6, [x0]\t\n"
      "add x0, x0, x1\t\n"
      "ldr d7, [x0]\t\n"

      "zip1 v16.2s, v0.2s, v1.2s\t\n"
      "zip1 v17.2s, v2.2s, v3.2s\t\n"
      "zip1 v18.2s, v4.2s, v5.2s\t\n"
      "zip1 v19.2s, v6.2s, v7.2s\t\n"

      "zip1 v20.2d, v16.2d, v17.2d\t\n"
      "zip1 v21.2d, v18.2d, v19.2d\t\n"
      "lsl x3, x3, #2\t\n"
      "stp q20, q21, [x2]\t\n"

      "zip2 v22.2s, v0.2s, v1.2s\t\n"
      "zip2 v23.2s, v2.2s, v3.2s\t\n"
      "zip2 v24.2s, v4.2s, v5.2s\t\n"
      "zip2 v25.2s, v6.2s, v7.2s\t\n"

      "zip1 v26.2d, v22.2d, v23.2d\t\n"
      "zip1 v27.2d, v24.2d, v25.2d\t\n"
      "add x2, x2, x3\t\n"
      "stp q26, q27, [x2]\t\n"

      :
      :
      [src] "r"(src), [ld_src] "r"(ld_src), [dst] "r"(dst), [ld_dst] "r"(ld_dst)
      : "memory",
        "cc",
        "x0",
        "x1",
        "x2",
        "x3",
        "v0",
        "v1",
        "v2",
        "v3",
        "v4",
        "v5",
        "v6",
        "v7",
        "v16",
        "v17",
        "v18",
        "v19",
        "v20",
        "v21",
        "v22",
        "v23",
        "v24",
        "v25",
        "v26",
        "v27");
}

static inline void transpose_kernel_4x8_neon(
    const float* src,
    int64_t ld_src,
    float* dst,
    int64_t ld_dst) {
  asm volatile(
      "mov x0, %[src]\t\n"
      "mov x1, %[ld_src]\t\n"
      "mov x2, %[dst]\t\n"
      "mov x3, %[ld_dst]\t\n"

      "ldp q0, q1, [x0]\t\n"
      "lsl x1, x1, #2\t\n"
      "add x0, x0, x1\t\n"
      "ldp q2, q3, [x0]\t\n"
      "add x0, x0, x1\t\n"
      "ldp q4, q5, [x0]\t\n"
      "add x0, x0, x1\t\n"
      "ldp q6, q7, [x0]\t\n"

      "zip1 v16.4s, v0.4s, v2.4s\t\n"
      "zip1 v17.4s, v4.4s, v6.4s\t\n"

      "zip1 v18.2d, v16.2d, v17.2d\t\n"
      "lsl x3, x3, #2\t\n"
      "str q18, [x2]\t\n"

      "zip2 v19.2d, v16.2d, v17.2d\t\n"
      "add x2, x2, x3\t\n"
      "str q19, [x2]\t\n"

      "zip2 v20.4s, v0.4s, v2.4s\t\n"
      "zip2 v21.4s, v4.4s, v6.4s\t\n"

      "zip1 v22.2d, v20.2d, v21.2d\t\n"
      "add x2, x2, x3\t\n"
      "str q22, [x2]\t\n"

      "zip2 v23.2d, v20.2d, v21.2d\t\n"
      "add x2, x2, x3\t\n"
      "str q23, [x2]\t\n"

      "zip1 v24.4s, v1.4s, v3.4s\t\n"
      "zip1 v25.4s, v5.4s, v7.4s\t\n"

      "zip1 v26.2d, v24.2d, v25.2d\t\n"
      "add x2, x2, x3\t\n"
      "str q26, [x2]\t\n"

      "zip2 v27.2d, v24.2d, v25.2d\t\n"
      "add x2, x2, x3\t\n"
      "str q27, [x2]\t\n"

      "zip2 v28.4s, v1.4s, v3.4s\t\n"
      "zip2 v29.4s, v5.4s, v7.4s\t\n"

      "zip1 v30.2d, v28.2d, v29.2d\t\n"
      "add x2, x2, x3\t\n"
      "str q30, [x2]\t\n"

      "zip2 v31.2d, v28.2d, v29.2d\t\n"
      "add x2, x2, x3\t\n"
      "str q31, [x2]\t\n"

      :
      :
      [src] "r"(src), [ld_src] "r"(ld_src), [dst] "r"(dst), [ld_dst] "r"(ld_dst)
      : "memory",
        "cc",
        "x0",
        "x1",
        "x2",
        "x3",
        "v0",
        "v1",
        "v2",
        "v3",
        "v4",
        "v5",
        "v6",
        "v7",
        "v16",
        "v17",
        "v18",
        "v19",
        "v20",
        "v21",
        "v22",
        "v23",
        "v24",
        "v25",
        "v26",
        "v27",
        "v28",
        "v29",
        "v30",
        "v31");
}

static inline void transpose_kernel_4x4_neon(
    const float* src,
    int64_t ld_src,
    float* dst,
    int64_t ld_dst) {
  asm volatile(
      "mov x0, %[src]\t\n"
      "mov x1, %[ld_src]\t\n"
      "mov x2, %[dst]\t\n"
      "mov x3, %[ld_dst]\t\n"

      "ldr q0, [x0]\t\n"
      "lsl x1, x1, #2\t\n"
      "add x0, x0, x1\t\n"
      "ldr q1, [x0]\t\n"
      "add x0, x0, x1\t\n"
      "ldr q2, [x0]\t\n"
      "add x0, x0, x1\t\n"
      "ldr q3, [x0]\t\n"

      "zip1 v4.4s, v0.4s, v1.4s\t\n"
      "zip1 v5.4s, v2.4s, v3.4s\t\n"

      "zip1 v6.2d, v4.2d, v5.2d\t\n"
      "lsl x3, x3, #2\t\n"
      "str q6, [x2]\t\n"

      "zip2 v7.2d, v4.2d, v5.2d\t\n"
      "add x2, x2, x3\t\n"
      "str q7, [x2]\t\n"

      "zip2 v16.4s, v0.4s, v1.4s\t\n"
      "zip2 v17.4s, v2.4s, v3.4s\t\n"

      "zip1 v18.2d, v16.2d, v17.2d\t\n"
      "add x2, x2, x3\t\n"
      "str q18, [x2]\t\n"

      "zip2 v19.2d, v16.2d, v17.2d\t\n"
      "add x2, x2, x3\t\n"
      "str q19, [x2]\t\n"

      :
      :
      [src] "r"(src), [ld_src] "r"(ld_src), [dst] "r"(dst), [ld_dst] "r"(ld_dst)
      : "memory",
        "cc",
        "x0",
        "x1",
        "x2",
        "x3",
        "v0",
        "v1",
        "v2",
        "v3",
        "v4",
        "v5",
        "v6",
        "v7",
        "v16",
        "v17",
        "v18",
        "v19");
}

static inline void transpose_kernel_4x2_neon(
    const float* src,
    int64_t ld_src,
    float* dst,
    int64_t ld_dst) {
  asm volatile(
      "mov x0, %[src]\t\n"
      "mov x1, %[ld_src]\t\n"
      "mov x2, %[dst]\t\n"
      "mov x3, %[ld_dst]\t\n"

      "ldr d0, [x0]\t\n"
      "lsl x1, x1, #2\t\n"
      "add x0, x0, x1\t\n"
      "ldr d1, [x0]\t\n"
      "add x0, x0, x1\t\n"
      "ldr d2, [x0]\t\n"
      "add x0, x0, x1\t\n"
      "ldr d3, [x0]\t\n"

      "zip1 v16.2s, v0.2s, v1.2s\t\n"
      "zip1 v17.2s, v2.2s, v3.2s\t\n"

      "zip1 v18.2d, v16.2d, v17.2d\t\n"
      "lsl x3, x3, #2\t\n"
      "str q18, [x2]\t\n"

      "zip2 v19.2s, v0.2s, v1.2s\t\n"
      "zip2 v20.2s, v2.2s, v3.2s\t\n"

      "zip1 v21.2d, v19.2d, v20.2d\t\n"
      "add x2, x2, x3\t\n"
      "str q21, [x2]\t\n"

      :
      :
      [src] "r"(src), [ld_src] "r"(ld_src), [dst] "r"(dst), [ld_dst] "r"(ld_dst)
      : "memory",
        "cc",
        "x0",
        "x1",
        "x2",
        "x3",
        "v0",
        "v1",
        "v2",
        "v3",
        "v4",
        "v5",
        "v6",
        "v7",
        "v16",
        "v17",
        "v18",
        "v19",
        "v20",
        "v21");
}

static inline void transpose_kernel_2x8_neon(
    const float* src,
    int64_t ld_src,
    float* dst,
    int64_t ld_dst) {
  asm volatile(
      "mov x0, %[src]\t\n"
      "mov x1, %[ld_src]\t\n"
      "mov x2, %[dst]\t\n"
      "mov x3, %[ld_dst]\t\n"

      "ldp q0, q1, [x0]\t\n"
      "lsl x1, x1, #2\t\n"
      "add x0, x0, x1\t\n"
      "ldp q2, q3, [x0]\t\n"

      "zip1 v4.4s, v0.4s, v2.4s\t\n"
      "lsl x3, x3, #2\t\n"
      "str d4, [x2]\t\n"

      "dup v5.2d, v4.d[1]\t\n"
      "add x2, x2, x3\t\n"
      "str d5, [x2]\t\n"

      "zip2 v6.4s, v0.4s, v2.4s\t\n"
      "add x2, x2, x3\t\n"
      "str d6, [x2]\t\n"

      "dup v7.2d, v6.d[1]\t\n"
      "add x2, x2, x3\t\n"
      "str d7, [x2]\t\n"

      "zip1 v16.4s, v1.4s, v3.4s\t\n"
      "add x2, x2, x3\t\n"
      "str d16, [x2]\t\n"

      "dup v17.2d, v16.d[1]\t\n"
      "add x2, x2, x3\t\n"
      "str d17, [x2]\t\n"

      "zip2 v18.4s, v1.4s, v3.4s\t\n"
      "add x2, x2, x3\t\n"
      "str d18, [x2]\t\n"

      "dup v19.2d, v18.d[1]\t\n"
      "add x2, x2, x3\t\n"
      "str d19, [x2]\t\n"

      :
      :
      [src] "r"(src), [ld_src] "r"(ld_src), [dst] "r"(dst), [ld_dst] "r"(ld_dst)
      : "memory",
        "cc",
        "x0",
        "x1",
        "x2",
        "x3",
        "v0",
        "v1",
        "v2",
        "v3",
        "v4",
        "v5",
        "v6",
        "v7",
        "v16",
        "v17",
        "v18",
        "v19");
}

static inline void transpose_kernel_2x4_neon(
    const float* src,
    int64_t ld_src,
    float* dst,
    int64_t ld_dst) {
  asm volatile(
      "mov x0, %[src]\t\n"
      "mov x1, %[ld_src]\t\n"
      "mov x2, %[dst]\t\n"
      "mov x3, %[ld_dst]\t\n"

      "ldr q0, [x0]\t\n"
      "lsl x1, x1, #2\t\n"
      "add x0, x0, x1\t\n"
      "ldr q1, [x0]\t\n"

      "zip1 v2.4s, v0.4s, v1.4s\t\n"

      "st1 {v2.d}[0], [x2]\t\n"
      "lsl x3, x3, #2\t\n"
      "add x2, x2, x3\t\n"
      "st1 {v2.d}[1], [x2]\t\n"

      "zip2 v3.4s, v0.4s, v1.4s\t\n"

      "add x2, x2, x3\t\n"
      "st1 {v3.d}[0], [x2]\t\n"
      "add x2, x2, x3\t\n"
      "st1 {v3.d}[1], [x2]\t\n"

      :
      :
      [src] "r"(src), [ld_src] "r"(ld_src), [dst] "r"(dst), [ld_dst] "r"(ld_dst)
      : "memory", "cc", "x0", "x1", "x2", "x3", "v0", "v1", "v2", "v3");
}

static inline void transpose_kernel_2x2_neon(
    const float* src,
    int64_t ld_src,
    float* dst,
    int64_t ld_dst) {
  asm volatile(
      "mov x0, %[src]\t\n"
      "mov x1, %[ld_src]\t\n"
      "mov x2, %[dst]\t\n"
      "mov x3, %[ld_dst]\t\n"

      "ldr d0, [x0]\t\n"
      "lsl x1, x1, #2\t\n"
      "add x0, x0, x1\t\n"
      "ldr d1, [x0]\t\n"

      "zip1 v2.4s, v0.4s, v1.4s\t\n"

      "st1 {v2.d}[0], [x2]\t\n"
      "lsl x3, x3, #2\t\n"
      "add x2, x2, x3\t\n"
      "st1 {v2.d}[1], [x2]\t\n"

      :
      :
      [src] "r"(src), [ld_src] "r"(ld_src), [dst] "r"(dst), [ld_dst] "r"(ld_dst)
      : "memory", "cc", "x0", "x1", "x2", "x3", "v0", "v1", "v2");
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
