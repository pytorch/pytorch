/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cpuinfo.h>
#include <gtest/gtest.h>

#include <qnnpack/isa-checks.h>
#include <qnnpack/q8gemm_sparse.h>

#include "gemm-block-sparse-microkernel-tester.h"

#define TEST_PACKED_ROW_BLOCK_SIZEXCOL_BLOCK_SIZE_SPARSE_OP(MR, \
    NR, row_block_size, col_block_size, \
    prepacking_kernel, compute_kernel) \
TEST(Q8GEMM__##MR ## x ##NR ## c##row_block_size ## x ##col_block_size__AARCH32_NEON, packedA_k_lt_4) { \
  TEST_REQUIRES_ARM_NEON; \
  GemmBlockSparseMicrokernelTester() \
  .mr(MR) \
  .nr(NR) \
  .m(MR) \
  .n(NR) \
  .k(3) \
  .rowBlockSize(row_block_size) \
  .colBlockSize(col_block_size) \
  .test_packed( \
      prepacking_kernel, \
      compute_kernel); \
} \
\
TEST(Q8GEMM__##MR ## x ##NR ## c##row_block_size ## x ##col_block_size__AARCH32_NEON, packedA_k_lt_4_strided_a) { \
  TEST_REQUIRES_ARM_NEON; \
  GemmBlockSparseMicrokernelTester() \
      .mr(MR) \
      .nr(NR) \
      .m(MR) \
      .n(NR) \
      .k(3) \
      .rowBlockSize(row_block_size) \
      .colBlockSize(col_block_size) \
      .aStride(37) \
      .test_packed( \
          prepacking_kernel, \
          compute_kernel); \
} \
 \
TEST(Q8GEMM__##MR ## x ##NR ## c##row_block_size ## x ##col_block_size__AARCH32_NEON, packedA_k_lt_4_strided_c) { \
  TEST_REQUIRES_ARM_NEON; \
  GemmBlockSparseMicrokernelTester() \
      .mr(MR) \
      .nr(NR) \
      .m(MR) \
      .n(NR) \
      .k(3) \
      .rowBlockSize(row_block_size) \
      .colBlockSize(col_block_size) \
      .cStride(17) \
      .test_packed( \
          prepacking_kernel, \
          compute_kernel); \
} \
 \
TEST(Q8GEMM__##MR ## x ##NR ## c##row_block_size ## x ##col_block_size__AARCH32_NEON, packedA_k_lt_4_qmin128) { \
  TEST_REQUIRES_ARM_NEON; \
  GemmBlockSparseMicrokernelTester() \
  .mr(MR) \
  .nr(NR) \
  .m(MR) \
  .n(NR) \
  .k(3) \
  .rowBlockSize(row_block_size) \
  .colBlockSize(col_block_size) \
  .qmin(128) \
  .test_packed( \
      prepacking_kernel, \
      compute_kernel); \
} \
 \
TEST(Q8GEMM__##MR ## x ##NR ## c##row_block_size ## x ##col_block_size__AARCH32_NEON, packedA_k_lt_4_qmax128) { \
  TEST_REQUIRES_ARM_NEON; \
  GemmBlockSparseMicrokernelTester() \
  .mr(MR) \
  .nr(NR) \
  .m(MR) \
  .n(NR) \
  .k(3) \
  .rowBlockSize(row_block_size) \
  .colBlockSize(col_block_size) \
  .qmax(128) \
  .test_packed( \
      prepacking_kernel, \
      compute_kernel); \
} \
 \
TEST(Q8GEMM__##MR ## x ##NR ## c##row_block_size ## x ##col_block_size__AARCH32_NEON, packedA_k_lt_4_azp0) { \
  TEST_REQUIRES_ARM_NEON; \
  GemmBlockSparseMicrokernelTester() \
      .mr(MR) \
      .nr(NR) \
      .m(MR) \
      .n(NR) \
      .k(3) \
      .rowBlockSize(row_block_size) \
      .colBlockSize(col_block_size) \
      .aZeroPoint(0) \
      .test_packed( \
          prepacking_kernel, \
          compute_kernel); \
} \
 \
TEST(Q8GEMM__##MR ## x ##NR ## c##row_block_size ## x ##col_block_size__AARCH32_NEON, packedA_k_lt_4_bzp0) { \
  TEST_REQUIRES_ARM_NEON; \
  GemmBlockSparseMicrokernelTester() \
      .mr(MR) \
      .nr(NR) \
      .m(MR) \
      .n(NR) \
      .k(3) \
      .rowBlockSize(row_block_size) \
      .colBlockSize(col_block_size) \
      .bZeroPoint(0) \
      .test_packed( \
          prepacking_kernel, \
          compute_kernel); \
} \
 \
TEST(Q8GEMM__##MR ## x ##NR ## c##row_block_size ## x ##col_block_size__AARCH32_NEON, packedA_k_lt_4_nozp) { \
  TEST_REQUIRES_ARM_NEON; \
  GemmBlockSparseMicrokernelTester() \
      .mr(MR) \
      .nr(NR) \
      .m(MR) \
      .n(NR) \
      .k(3) \
      .rowBlockSize(row_block_size) \
      .colBlockSize(col_block_size) \
      .aZeroPoint(0) \
      .bZeroPoint(0) \
      .test_packed( \
          prepacking_kernel, \
          compute_kernel); \
} \
 \
TEST(Q8GEMM__##MR ## x ##NR ## c##row_block_size ## x ##col_block_size__AARCH32_NEON, packedA_k_lt_8) { \
  TEST_REQUIRES_ARM_NEON; \
  GemmBlockSparseMicrokernelTester() \
  .mr(MR) \
  .nr(NR) \
  .m(MR) \
  .n(NR) \
  .k(5) \
  .rowBlockSize(row_block_size) \
  .colBlockSize(col_block_size) \
  .test_packed( \
      prepacking_kernel, \
      compute_kernel); \
} \
 \
TEST(Q8GEMM__##MR ## x ##NR ## c##row_block_size ## x ##col_block_size__AARCH32_NEON, packedA_k_lt_8_strided_a) { \
  TEST_REQUIRES_ARM_NEON; \
  GemmBlockSparseMicrokernelTester() \
      .mr(MR) \
      .nr(NR) \
      .m(MR) \
      .n(NR) \
      .k(5) \
      .rowBlockSize(row_block_size) \
      .colBlockSize(col_block_size) \
      .aStride(37) \
      .test_packed( \
          prepacking_kernel, \
          compute_kernel); \
} \
 \
TEST(Q8GEMM__##MR ## x ##NR ## c##row_block_size ## x ##col_block_size__AARCH32_NEON, packedA_k_lt_8_strided_c) { \
  TEST_REQUIRES_ARM_NEON; \
  GemmBlockSparseMicrokernelTester() \
      .mr(MR) \
      .nr(NR) \
      .m(MR) \
      .n(NR) \
      .k(5) \
      .rowBlockSize(row_block_size) \
      .colBlockSize(col_block_size) \
      .cStride(17) \
      .test_packed( \
          prepacking_kernel, \
          compute_kernel); \
} \
 \
TEST(Q8GEMM__##MR ## x ##NR ## c##row_block_size ## x ##col_block_size__AARCH32_NEON, packedA_k_lt_8_qmin128) { \
  TEST_REQUIRES_ARM_NEON; \
  GemmBlockSparseMicrokernelTester() \
  .mr(MR) \
  .nr(NR) \
  .m(MR) \
  .n(NR) \
  .k(5) \
  .rowBlockSize(row_block_size) \
  .colBlockSize(col_block_size) \
  .qmin(128) \
  .test_packed( \
      prepacking_kernel, \
      compute_kernel); \
} \
 \
TEST(Q8GEMM__##MR ## x ##NR ## c##row_block_size ## x ##col_block_size__AARCH32_NEON, packedA_k_lt_8_qmax128) { \
  TEST_REQUIRES_ARM_NEON; \
  GemmBlockSparseMicrokernelTester() \
  .mr(MR) \
  .nr(NR) \
  .m(MR) \
  .n(NR) \
  .k(5) \
  .rowBlockSize(row_block_size) \
  .colBlockSize(col_block_size) \
  .qmax(128) \
  .test_packed( \
      prepacking_kernel, \
      compute_kernel); \
} \
 \
TEST(Q8GEMM__##MR ## x ##NR ## c##row_block_size ## x ##col_block_size__AARCH32_NEON, packedA_k_lt_8_azp0) { \
  TEST_REQUIRES_ARM_NEON; \
  GemmBlockSparseMicrokernelTester() \
      .mr(MR) \
      .nr(NR) \
      .m(MR) \
      .n(NR) \
      .k(5) \
      .rowBlockSize(row_block_size) \
      .colBlockSize(col_block_size) \
      .aZeroPoint(0) \
      .test_packed( \
          prepacking_kernel, \
          compute_kernel); \
} \
 \
TEST(Q8GEMM__##MR ## x ##NR ## c##row_block_size ## x ##col_block_size__AARCH32_NEON, packedA_k_lt_8_bzp0) { \
  TEST_REQUIRES_ARM_NEON; \
  GemmBlockSparseMicrokernelTester() \
      .mr(MR) \
      .nr(NR) \
      .m(MR) \
      .n(NR) \
      .k(5) \
      .rowBlockSize(row_block_size) \
      .colBlockSize(col_block_size) \
      .bZeroPoint(0) \
      .test_packed( \
          prepacking_kernel, \
          compute_kernel); \
} \
 \
TEST(Q8GEMM__##MR ## x ##NR ## c##row_block_size ## x ##col_block_size__AARCH32_NEON, packedA_k_lt_8_nozp) { \
  TEST_REQUIRES_ARM_NEON; \
  GemmBlockSparseMicrokernelTester() \
      .mr(MR) \
      .nr(NR) \
      .m(MR) \
      .n(NR) \
      .k(5) \
      .rowBlockSize(row_block_size) \
      .colBlockSize(col_block_size) \
      .aZeroPoint(0) \
      .bZeroPoint(0) \
      .test_packed( \
          prepacking_kernel, \
          compute_kernel); \
} \
 \
TEST(Q8GEMM__##MR ## x ##NR ## c##row_block_size ## x ##col_block_size__AARCH32_NEON, packedA_k_eq_8) { \
  TEST_REQUIRES_ARM_NEON; \
  GemmBlockSparseMicrokernelTester() \
  .mr(MR) \
  .nr(NR) \
  .m(MR) \
  .n(NR) \
  .k(8) \
  .rowBlockSize(row_block_size) \
  .colBlockSize(col_block_size) \
  .test_packed( \
      prepacking_kernel, \
      compute_kernel); \
} \
 \
TEST(Q8GEMM__##MR ## x ##NR ## c##row_block_size ## x ##col_block_size__AARCH32_NEON, packedA_k_eq_8_strided_a) { \
  TEST_REQUIRES_ARM_NEON; \
  GemmBlockSparseMicrokernelTester() \
      .mr(MR) \
      .nr(NR) \
      .m(MR) \
      .n(NR) \
      .k(8) \
      .rowBlockSize(row_block_size) \
      .colBlockSize(col_block_size) \
      .aStride(37) \
      .test_packed( \
          prepacking_kernel, \
          compute_kernel); \
} \
 \
TEST(Q8GEMM__##MR ## x ##NR ## c##row_block_size ## x ##col_block_size__AARCH32_NEON, packedA_k_eq_8_strided_c) { \
  TEST_REQUIRES_ARM_NEON; \
  GemmBlockSparseMicrokernelTester() \
      .mr(MR) \
      .nr(NR) \
      .m(MR) \
      .n(NR) \
      .k(8) \
      .rowBlockSize(row_block_size) \
      .colBlockSize(col_block_size) \
      .cStride(17) \
      .test_packed( \
          prepacking_kernel, \
          compute_kernel); \
} \
 \
TEST(Q8GEMM__##MR ## x ##NR ## c##row_block_size ## x ##col_block_size__AARCH32_NEON, packedA_k_eq_8_qmin128) { \
  TEST_REQUIRES_ARM_NEON; \
  GemmBlockSparseMicrokernelTester() \
  .mr(MR) \
  .nr(NR) \
  .m(MR) \
  .n(NR) \
  .k(8) \
  .rowBlockSize(row_block_size) \
  .colBlockSize(col_block_size) \
  .qmin(128) \
  .test_packed( \
      prepacking_kernel, \
      compute_kernel); \
} \
 \
TEST(Q8GEMM__##MR ## x ##NR ## c##row_block_size ## x ##col_block_size__AARCH32_NEON, packedA_k_eq_8_qmax128) { \
  TEST_REQUIRES_ARM_NEON; \
  GemmBlockSparseMicrokernelTester() \
  .mr(MR) \
  .nr(NR) \
  .m(MR) \
  .n(NR) \
  .k(8) \
  .rowBlockSize(row_block_size) \
  .colBlockSize(col_block_size) \
  .qmax(128) \
  .test_packed( \
      prepacking_kernel, \
      compute_kernel); \
} \
 \
TEST(Q8GEMM__##MR ## x ##NR ## c##row_block_size ## x ##col_block_size__AARCH32_NEON, packedA_k_eq_8_azp0) { \
  TEST_REQUIRES_ARM_NEON; \
  GemmBlockSparseMicrokernelTester() \
      .mr(MR) \
      .nr(NR) \
      .m(MR) \
      .n(NR) \
      .k(8) \
      .rowBlockSize(row_block_size) \
      .colBlockSize(col_block_size) \
      .aZeroPoint(0) \
      .test_packed( \
          prepacking_kernel, \
          compute_kernel); \
} \
 \
TEST(Q8GEMM__##MR ## x ##NR ## c##row_block_size ## x ##col_block_size__AARCH32_NEON, packedA_k_eq_8_bzp0) { \
  TEST_REQUIRES_ARM_NEON; \
  GemmBlockSparseMicrokernelTester() \
      .mr(MR) \
      .nr(NR) \
      .m(MR) \
      .n(NR) \
      .k(8) \
      .rowBlockSize(row_block_size) \
      .colBlockSize(col_block_size) \
      .bZeroPoint(0) \
      .test_packed( \
          prepacking_kernel, \
          compute_kernel); \
} \
 \
TEST(Q8GEMM__##MR ## x ##NR ## c##row_block_size ## x ##col_block_size__AARCH32_NEON, packedA_k_eq_8_nozp) { \
  TEST_REQUIRES_ARM_NEON; \
  GemmBlockSparseMicrokernelTester() \
      .mr(MR) \
      .nr(NR) \
      .m(MR) \
      .n(NR) \
      .k(8) \
      .rowBlockSize(row_block_size) \
      .colBlockSize(col_block_size) \
      .aZeroPoint(0) \
      .bZeroPoint(0) \
      .test_packed( \
          prepacking_kernel, \
          compute_kernel); \
} \
 \
TEST(Q8GEMM__##MR ## x ##NR ## c##row_block_size ## x ##col_block_size__AARCH32_NEON, packedA_k_gt_8) { \
  TEST_REQUIRES_ARM_NEON; \
  for (size_t k = 9; k < 16; k++) { \
    GemmBlockSparseMicrokernelTester() \
    .mr(MR) \
    .nr(NR) \
    .m(MR) \
    .n(NR) \
    .k(k) \
    .rowBlockSize(row_block_size) \
    .colBlockSize(col_block_size) \
    .test_packed( \
        prepacking_kernel, \
        compute_kernel); \
  } \
} \
 \
TEST(Q8GEMM__##MR ## x ##NR ## c##row_block_size ## x ##col_block_size__AARCH32_NEON, packedA_k_gt_8_strided_a) { \
  TEST_REQUIRES_ARM_NEON; \
  for (size_t k = 9; k < 16; k++) { \
    GemmBlockSparseMicrokernelTester() \
        .mr(MR) \
        .nr(NR) \
        .m(MR) \
        .n(NR) \
        .k(k) \
        .rowBlockSize(row_block_size) \
        .colBlockSize(col_block_size) \
        .aStride(37) \
        .test_packed( \
            prepacking_kernel, \
            compute_kernel); \
  } \
} \
 \
TEST(Q8GEMM__##MR ## x ##NR ## c##row_block_size ## x ##col_block_size__AARCH32_NEON, packedA_k_gt_8_strided_c) { \
  TEST_REQUIRES_ARM_NEON; \
  for (size_t k = 9; k < 16; k++) { \
    GemmBlockSparseMicrokernelTester() \
        .mr(MR) \
        .nr(NR) \
        .m(MR) \
        .n(NR) \
        .k(k) \
        .rowBlockSize(row_block_size) \
        .colBlockSize(col_block_size) \
        .cStride(17) \
        .test_packed( \
            prepacking_kernel, \
            compute_kernel); \
  } \
} \
 \
TEST(Q8GEMM__##MR ## x ##NR ## c##row_block_size ## x ##col_block_size__AARCH32_NEON, packedA_k_gt_8_azp0) { \
  TEST_REQUIRES_ARM_NEON; \
  for (size_t k = 9; k < 16; k++) { \
    GemmBlockSparseMicrokernelTester() \
        .mr(MR) \
        .nr(NR) \
        .m(MR) \
        .n(NR) \
        .k(k) \
        .rowBlockSize(row_block_size) \
        .colBlockSize(col_block_size) \
        .aZeroPoint(0) \
        .test_packed( \
            prepacking_kernel, \
            compute_kernel); \
  } \
} \
 \
TEST(Q8GEMM__##MR ## x ##NR ## c##row_block_size ## x ##col_block_size__AARCH32_NEON, packedA_k_gt_8_bzp0) { \
  TEST_REQUIRES_ARM_NEON; \
  for (size_t k = 9; k < 16; k++) { \
    GemmBlockSparseMicrokernelTester() \
        .mr(MR) \
        .nr(NR) \
        .m(MR) \
        .n(NR) \
        .k(k) \
        .rowBlockSize(row_block_size) \
        .colBlockSize(col_block_size) \
        .bZeroPoint(0) \
        .test_packed( \
            prepacking_kernel, \
            compute_kernel); \
  } \
} \
 \
TEST(Q8GEMM__##MR ## x ##NR ## c##row_block_size ## x ##col_block_size__AARCH32_NEON, packedA_k_gt_8_nozp) { \
  TEST_REQUIRES_ARM_NEON; \
  for (size_t k = 9; k < 16; k++) { \
    GemmBlockSparseMicrokernelTester() \
        .mr(MR) \
        .nr(NR) \
        .m(MR) \
        .n(NR) \
        .k(k) \
        .rowBlockSize(row_block_size) \
        .colBlockSize(col_block_size) \
        .aZeroPoint(0) \
        .bZeroPoint(0) \
        .test_packed( \
            prepacking_kernel, \
            compute_kernel); \
  } \
} \
 \
TEST(Q8GEMM__##MR ## x ##NR ## c##row_block_size ## x ##col_block_size__AARCH32_NEON, packedA_k_gt_8_subtile) { \
  TEST_REQUIRES_ARM_NEON; \
  for (size_t k = 9; k < 16; k++) { \
    for (uint32_t m = 1; m <= MR; m++) { \
      for (uint32_t n = 1; n <= NR; n++) { \
        GemmBlockSparseMicrokernelTester() \
            .mr(MR) \
            .nr(NR) \
            .m(m) \
            .n(n) \
            .k(k) \
            .rowBlockSize(row_block_size) \
            .colBlockSize(col_block_size) \
            .iterations(3) \
            .test_packed( \
                prepacking_kernel, \
                compute_kernel); \
      } \
    } \
  } \
} \
 \
TEST(Q8GEMM__##MR ## x ##NR ## c##row_block_size ## x ##col_block_size__AARCH32_NEON, packedA_k_div_8) { \
  TEST_REQUIRES_ARM_NEON; \
  for (size_t k = 16; k < 128; k += 8) { \
    GemmBlockSparseMicrokernelTester() \
    .mr(MR) \
    .nr(NR) \
    .m(MR) \
    .n(NR) \
    .k(k) \
    .rowBlockSize(row_block_size) \
    .colBlockSize(col_block_size) \
    .test_packed( \
        prepacking_kernel, \
        compute_kernel); \
  } \
} \
 \
TEST(Q8GEMM__##MR ## x ##NR ## c##row_block_size ## x ##col_block_size__AARCH32_NEON, packedA_k_div_8_strided_a) { \
  TEST_REQUIRES_ARM_NEON; \
  for (size_t k = 16; k < 128; k += 8) { \
    GemmBlockSparseMicrokernelTester() \
        .mr(MR) \
        .nr(NR) \
        .m(MR) \
        .n(NR) \
        .k(k) \
        .rowBlockSize(row_block_size) \
        .colBlockSize(col_block_size) \
        .aStride(171) \
        .test_packed( \
            prepacking_kernel, \
            compute_kernel); \
  } \
} \
 \
TEST(Q8GEMM__##MR ## x ##NR ## c##row_block_size ## x ##col_block_size__AARCH32_NEON, packedA_k_div_8_strided_c) { \
  TEST_REQUIRES_ARM_NEON; \
  for (size_t k = 16; k < 128; k += 8) { \
    GemmBlockSparseMicrokernelTester() \
        .mr(MR) \
        .nr(NR) \
        .m(MR) \
        .n(NR) \
        .k(k) \
        .rowBlockSize(row_block_size) \
        .colBlockSize(col_block_size) \
        .cStride(17) \
        .test_packed( \
            prepacking_kernel, \
            compute_kernel); \
  } \
} \
 \
TEST(Q8GEMM__##MR ## x ##NR ## c##row_block_size ## x ##col_block_size__AARCH32_NEON, packedA_k_div_8_subtile) { \
  TEST_REQUIRES_ARM_NEON; \
  for (size_t k = 16; k < 128; k += 24) { \
    for (uint32_t m = 1; m <= MR; m++) { \
      for (uint32_t n = 1; n <= NR; n++) { \
        GemmBlockSparseMicrokernelTester() \
            .mr(MR) \
            .nr(NR) \
            .m(m) \
            .n(n) \
            .k(k) \
            .rowBlockSize(row_block_size) \
            .colBlockSize(col_block_size) \
            .iterations(3) \
            .test_packed( \
                prepacking_kernel, \
                compute_kernel); \
      } \
    } \
  } \
}

#define TEST_PACKED_1x4_SPARSE_OP(MR, NR, prepacking_kernel, compute_kernel) \
  TEST_PACKED_ROW_BLOCK_SIZEXCOL_BLOCK_SIZE_SPARSE_OP(MR, \
      NR, 1, 4, prepacking_kernel, compute_kernel)
#define TEST_PACKED_8x1_SPARSE_OP(MR, NR, prepacking_kernel, compute_kernel) \
  TEST_PACKED_ROW_BLOCK_SIZEXCOL_BLOCK_SIZE_SPARSE_OP(MR, \
      NR, 8, 1, prepacking_kernel, compute_kernel)

#if CPUINFO_ARCH_ARM
TEST_PACKED_1x4_SPARSE_OP(
    8,
    4,
    pytorch_q8gemm_sparse_packA_ukernel_8x4__aarch32_neon,
    pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA__aarch32_neon)
TEST_PACKED_1x4_SPARSE_OP(
    4,
    8,
    pytorch_q8gemm_sparse_packA_ukernel_4x4__aarch32_neon,
    pytorch_q8gemm_dq_sparse_1x4_ukernel_4x8_packedA__aarch32_neon)
TEST_PACKED_8x1_SPARSE_OP(
    4,
    8,
    pytorch_q8gemm_sparse_packA_ukernel_4x4__aarch32_neon,
    pytorch_q8gemm_dq_sparse_8x1_ukernel_4x8_packedA__aarch32_neon)

#endif

#if CPUINFO_ARCH_ARM64

TEST_PACKED_1x4_SPARSE_OP(
    8,
    8,
    pytorch_q8gemm_sparse_packA_ukernel_8x4__aarch64_neon,
    pytorch_q8gemm_dq_sparse_1x4_ukernel_8x8_packedA__aarch64_neon)

#endif

#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64

TEST(Q8GEMM_8x4c1x4__SSE2, k_lt_4) {
  TEST_REQUIRES_X86_SSE2;
  GemmBlockSparseMicrokernelTester().mr(8).nr(4).m(8).n(4).k(3).test(
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4__sse2);
}

TEST(Q8GEMM_8x4c1x4__SSE2, k_lt_4_strided_a) {
  TEST_REQUIRES_X86_SSE2;
  GemmBlockSparseMicrokernelTester()
      .mr(8)
      .nr(4)
      .m(8)
      .n(4)
      .k(3)
      .aStride(37)
      .test(pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4__sse2);
}

TEST(Q8GEMM_8x4c1x4__SSE2, k_lt_4_strided_c) {
  TEST_REQUIRES_X86_SSE2;
  GemmBlockSparseMicrokernelTester()
      .mr(8)
      .nr(4)
      .m(8)
      .n(4)
      .k(3)
      .cStride(17)
      .test(pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4__sse2);
}

TEST(Q8GEMM_8x4c1x4__SSE2, k_lt_4_qmin128) {
  TEST_REQUIRES_X86_SSE2;
  GemmBlockSparseMicrokernelTester().mr(8).nr(4).m(8).n(4).k(3).qmin(128).test(
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4__sse2);
}

TEST(Q8GEMM_8x4c1x4__SSE2, k_lt_4_qmax128) {
  TEST_REQUIRES_X86_SSE2;
  GemmBlockSparseMicrokernelTester().mr(8).nr(4).m(8).n(4).k(3).qmax(128).test(
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4__sse2);
}

TEST(Q8GEMM_8x4c1x4__SSE2, k_lt_4_azp0) {
  TEST_REQUIRES_X86_SSE2;
  GemmBlockSparseMicrokernelTester()
      .mr(8)
      .nr(4)
      .m(8)
      .n(4)
      .k(3)
      .aZeroPoint(0)
      .test(pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4__sse2);
}

TEST(Q8GEMM_8x4c1x4__SSE2, k_lt_4_bzp0) {
  TEST_REQUIRES_X86_SSE2;
  GemmBlockSparseMicrokernelTester()
      .mr(8)
      .nr(4)
      .m(8)
      .n(4)
      .k(3)
      .bZeroPoint(0)
      .test(pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4__sse2);
}

TEST(Q8GEMM_8x4c1x4__SSE2, k_lt_4_nozp) {
  TEST_REQUIRES_X86_SSE2;
  GemmBlockSparseMicrokernelTester()
      .mr(8)
      .nr(4)
      .m(8)
      .n(4)
      .k(3)
      .aZeroPoint(0)
      .bZeroPoint(0)
      .test(pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4__sse2);
}

TEST(Q8GEMM_8x4c1x4__SSE2, k_lt_8) {
  TEST_REQUIRES_X86_SSE2;
  GemmBlockSparseMicrokernelTester().mr(8).nr(4).m(8).n(4).k(5).test(
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4__sse2);
}

TEST(Q8GEMM_8x4c1x4__SSE2, k_lt_8_strided_a) {
  TEST_REQUIRES_X86_SSE2;
  GemmBlockSparseMicrokernelTester()
      .mr(8)
      .nr(4)
      .m(8)
      .n(4)
      .k(5)
      .aStride(37)
      .test(pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4__sse2);
}

TEST(Q8GEMM_8x4c1x4__SSE2, k_lt_8_strided_c) {
  TEST_REQUIRES_X86_SSE2;
  GemmBlockSparseMicrokernelTester()
      .mr(8)
      .nr(4)
      .m(8)
      .n(4)
      .k(5)
      .cStride(17)
      .test(pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4__sse2);
}

TEST(Q8GEMM_8x4c1x4__SSE2, k_lt_8_qmin128) {
  TEST_REQUIRES_X86_SSE2;
  GemmBlockSparseMicrokernelTester().mr(8).nr(4).m(8).n(4).k(5).qmin(128).test(
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4__sse2);
}

TEST(Q8GEMM_8x4c1x4__SSE2, k_lt_8_qmax128) {
  TEST_REQUIRES_X86_SSE2;
  GemmBlockSparseMicrokernelTester().mr(8).nr(4).m(8).n(4).k(5).qmax(128).test(
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4__sse2);
}

TEST(Q8GEMM_8x4c1x4__SSE2, k_lt_8_azp0) {
  TEST_REQUIRES_X86_SSE2;
  GemmBlockSparseMicrokernelTester()
      .mr(8)
      .nr(4)
      .m(8)
      .n(4)
      .k(5)
      .aZeroPoint(0)
      .test(pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4__sse2);
}

TEST(Q8GEMM_8x4c1x4__SSE2, k_lt_8_bzp0) {
  TEST_REQUIRES_X86_SSE2;
  GemmBlockSparseMicrokernelTester()
      .mr(8)
      .nr(4)
      .m(8)
      .n(4)
      .k(5)
      .bZeroPoint(0)
      .test(pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4__sse2);
}

TEST(Q8GEMM_8x4c1x4__SSE2, k_lt_8_nozp) {
  TEST_REQUIRES_X86_SSE2;
  GemmBlockSparseMicrokernelTester()
      .mr(8)
      .nr(4)
      .m(8)
      .n(4)
      .k(5)
      .aZeroPoint(0)
      .bZeroPoint(0)
      .test(pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4__sse2);
}

TEST(Q8GEMM_8x4c1x4__SSE2, k_eq_8) {
  TEST_REQUIRES_X86_SSE2;
  GemmBlockSparseMicrokernelTester().mr(8).nr(4).m(8).n(4).k(8).test(
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4__sse2);
}

TEST(Q8GEMM_8x4c1x4__SSE2, k_eq_8_strided_a) {
  TEST_REQUIRES_X86_SSE2;
  GemmBlockSparseMicrokernelTester()
      .mr(8)
      .nr(4)
      .m(8)
      .n(4)
      .k(8)
      .aStride(37)
      .test(pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4__sse2);
}

TEST(Q8GEMM_8x4c1x4__SSE2, k_eq_8_strided_c) {
  TEST_REQUIRES_X86_SSE2;
  GemmBlockSparseMicrokernelTester()
      .mr(8)
      .nr(4)
      .m(8)
      .n(4)
      .k(8)
      .cStride(17)
      .test(pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4__sse2);
}

TEST(Q8GEMM_8x4c1x4__SSE2, k_eq_8_qmin128) {
  TEST_REQUIRES_X86_SSE2;
  GemmBlockSparseMicrokernelTester().mr(8).nr(4).m(8).n(4).k(8).qmin(128).test(
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4__sse2);
}

TEST(Q8GEMM_8x4c1x4__SSE2, k_eq_8_qmax128) {
  TEST_REQUIRES_X86_SSE2;
  GemmBlockSparseMicrokernelTester().mr(8).nr(4).m(8).n(4).k(8).qmax(128).test(
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4__sse2);
}

TEST(Q8GEMM_8x4c1x4__SSE2, k_eq_8_azp0) {
  TEST_REQUIRES_X86_SSE2;
  GemmBlockSparseMicrokernelTester()
      .mr(8)
      .nr(4)
      .m(8)
      .n(4)
      .k(8)
      .aZeroPoint(0)
      .test(pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4__sse2);
}

TEST(Q8GEMM_8x4c1x4__SSE2, k_eq_8_bzp0) {
  TEST_REQUIRES_X86_SSE2;
  GemmBlockSparseMicrokernelTester()
      .mr(8)
      .nr(4)
      .m(8)
      .n(4)
      .k(8)
      .bZeroPoint(0)
      .test(pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4__sse2);
}

TEST(Q8GEMM_8x4c1x4__SSE2, k_eq_8_nozp) {
  TEST_REQUIRES_X86_SSE2;
  GemmBlockSparseMicrokernelTester()
      .mr(8)
      .nr(4)
      .m(8)
      .n(4)
      .k(8)
      .aZeroPoint(0)
      .bZeroPoint(0)
      .test(pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4__sse2);
}

TEST(Q8GEMM_8x4c1x4__SSE2, k_gt_8) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t k = 9; k < 16; k++) {
    GemmBlockSparseMicrokernelTester().mr(8).nr(4).m(8).n(4).k(k).test(
        pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4__sse2);
  }
}

TEST(Q8GEMM_8x4c1x4__SSE2, k_gt_8_strided_a) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t k = 9; k < 16; k++) {
    GemmBlockSparseMicrokernelTester()
        .mr(8)
        .nr(4)
        .m(8)
        .n(4)
        .k(k)
        .aStride(37)
        .test(pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4__sse2);
  }
}

TEST(Q8GEMM_8x4c1x4__SSE2, k_gt_8_strided_c) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t k = 9; k < 16; k++) {
    GemmBlockSparseMicrokernelTester()
        .mr(8)
        .nr(4)
        .m(8)
        .n(4)
        .k(k)
        .cStride(17)
        .test(pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4__sse2);
  }
}

TEST(Q8GEMM_8x4c1x4__SSE2, k_gt_8_azp0) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t k = 9; k < 16; k++) {
    GemmBlockSparseMicrokernelTester()
        .mr(8)
        .nr(4)
        .m(8)
        .n(4)
        .k(k)
        .aZeroPoint(0)
        .test(pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4__sse2);
  }
}

TEST(Q8GEMM_8x4c1x4__SSE2, k_gt_8_bzp0) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t k = 9; k < 16; k++) {
    GemmBlockSparseMicrokernelTester()
        .mr(8)
        .nr(4)
        .m(8)
        .n(4)
        .k(k)
        .bZeroPoint(0)
        .test(pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4__sse2);
  }
}

TEST(Q8GEMM_8x4c1x4__SSE2, k_gt_8_nozp) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t k = 9; k < 16; k++) {
    GemmBlockSparseMicrokernelTester()
        .mr(8)
        .nr(4)
        .m(8)
        .n(4)
        .k(k)
        .aZeroPoint(0)
        .bZeroPoint(0)
        .test(pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4__sse2);
  }
}

TEST(Q8GEMM_8x4c1x4__SSE2, k_gt_8_subtile) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t k = 9; k < 16; k++) {
    for (uint32_t m = 1; m <= 8; m++) {
      for (uint32_t n = 1; n <= 4; n++) {
        GemmBlockSparseMicrokernelTester()
            .mr(8)
            .nr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(3)
            .test(pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4__sse2);
      }
    }
  }
}

TEST(Q8GEMM_8x4c1x4__SSE2, k_div_8) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t k = 16; k < 128; k += 8) {
    GemmBlockSparseMicrokernelTester().mr(8).nr(4).m(8).n(4).k(k).test(
        pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4__sse2);
  }
}

TEST(Q8GEMM_8x4c1x4__SSE2, k_div_8_strided_a) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t k = 16; k < 128; k += 8) {
    GemmBlockSparseMicrokernelTester()
        .mr(8)
        .nr(4)
        .m(8)
        .n(4)
        .k(k)
        .aStride(171)
        .test(pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4__sse2);
  }
}

TEST(Q8GEMM_8x4c1x4__SSE2, k_div_8_strided_c) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t k = 16; k < 128; k += 8) {
    GemmBlockSparseMicrokernelTester()
        .mr(8)
        .nr(4)
        .m(8)
        .n(4)
        .k(k)
        .cStride(17)
        .test(pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4__sse2);
  }
}

TEST(Q8GEMM_8x4c1x4__SSE2, k_div_8_subtile) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t k = 16; k < 128; k += 24) {
    for (uint32_t m = 1; m <= 8; m++) {
      for (uint32_t n = 1; n <= 4; n++) {
        GemmBlockSparseMicrokernelTester()
            .mr(8)
            .nr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(3)
            .test(pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4__sse2);
      }
    }
  }
}

TEST(Q8GEMM_8x4c1x4__SSE2, packedA_k_lt_4) {
  TEST_REQUIRES_X86_SSE2;
  GemmBlockSparseMicrokernelTester().mr(8).nr(4).m(8).n(4).k(3).test_packed(
      pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA__sse2);
}

TEST(Q8GEMM_8x4c1x4__SSE2, packedA_k_lt_4_strided_a) {
  TEST_REQUIRES_X86_SSE2;
  GemmBlockSparseMicrokernelTester()
      .mr(8)
      .nr(4)
      .m(8)
      .n(4)
      .k(3)
      .aStride(37)
      .test_packed(
          pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
          pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA__sse2);
}

TEST(Q8GEMM_8x4c1x4__SSE2, packedA_k_lt_4_strided_c) {
  TEST_REQUIRES_X86_SSE2;
  GemmBlockSparseMicrokernelTester()
      .mr(8)
      .nr(4)
      .m(8)
      .n(4)
      .k(3)
      .cStride(17)
      .test_packed(
          pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
          pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA__sse2);
}

TEST(Q8GEMM_8x4c1x4__SSE2, packedA_k_lt_4_qmin128) {
  TEST_REQUIRES_X86_SSE2;
  GemmBlockSparseMicrokernelTester().mr(8).nr(4).m(8).n(4).k(3).qmin(128).test_packed(
      pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA__sse2);
}

TEST(Q8GEMM_8x4c1x4__SSE2, packedA_k_lt_4_qmax128) {
  TEST_REQUIRES_X86_SSE2;
  GemmBlockSparseMicrokernelTester().mr(8).nr(4).m(8).n(4).k(3).qmax(128).test_packed(
      pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA__sse2);
}

TEST(Q8GEMM_8x4c1x4__SSE2, packedA_k_lt_4_azp0) {
  TEST_REQUIRES_X86_SSE2;
  GemmBlockSparseMicrokernelTester()
      .mr(8)
      .nr(4)
      .m(8)
      .n(4)
      .k(3)
      .aZeroPoint(0)
      .test_packed(
          pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
          pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA__sse2);
}

TEST(Q8GEMM_8x4c1x4__SSE2, packedA_k_lt_4_bzp0) {
  TEST_REQUIRES_X86_SSE2;
  GemmBlockSparseMicrokernelTester()
      .mr(8)
      .nr(4)
      .m(8)
      .n(4)
      .k(3)
      .bZeroPoint(0)
      .test_packed(
          pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
          pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA__sse2);
}

TEST(Q8GEMM_8x4c1x4__SSE2, packedA_k_lt_4_nozp) {
  TEST_REQUIRES_X86_SSE2;
  GemmBlockSparseMicrokernelTester()
      .mr(8)
      .nr(4)
      .m(8)
      .n(4)
      .k(3)
      .aZeroPoint(0)
      .bZeroPoint(0)
      .test_packed(
          pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
          pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA__sse2);
}

TEST(Q8GEMM_8x4c1x4__SSE2, packedA_k_lt_8) {
  TEST_REQUIRES_X86_SSE2;
  GemmBlockSparseMicrokernelTester().mr(8).nr(4).m(8).n(4).k(5).test_packed(
      pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA__sse2);
}

TEST(Q8GEMM_8x4c1x4__SSE2, packedA_k_lt_8_strided_a) {
  TEST_REQUIRES_X86_SSE2;
  GemmBlockSparseMicrokernelTester()
      .mr(8)
      .nr(4)
      .m(8)
      .n(4)
      .k(5)
      .aStride(37)
      .test_packed(
          pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
          pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA__sse2);
}

TEST(Q8GEMM_8x4c1x4__SSE2, packedA_k_lt_8_strided_c) {
  TEST_REQUIRES_X86_SSE2;
  GemmBlockSparseMicrokernelTester()
      .mr(8)
      .nr(4)
      .m(8)
      .n(4)
      .k(5)
      .cStride(17)
      .test_packed(
          pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
          pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA__sse2);
}

TEST(Q8GEMM_8x4c1x4__SSE2, packedA_k_lt_8_qmin128) {
  TEST_REQUIRES_X86_SSE2;
  GemmBlockSparseMicrokernelTester().mr(8).nr(4).m(8).n(4).k(5).qmin(128).test_packed(
      pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA__sse2);
}

TEST(Q8GEMM_8x4c1x4__SSE2, packedA_k_lt_8_qmax128) {
  TEST_REQUIRES_X86_SSE2;
  GemmBlockSparseMicrokernelTester().mr(8).nr(4).m(8).n(4).k(5).qmax(128).test_packed(
      pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA__sse2);
}

TEST(Q8GEMM_8x4c1x4__SSE2, packedA_k_lt_8_azp0) {
  TEST_REQUIRES_X86_SSE2;
  GemmBlockSparseMicrokernelTester()
      .mr(8)
      .nr(4)
      .m(8)
      .n(4)
      .k(5)
      .aZeroPoint(0)
      .test_packed(
          pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
          pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA__sse2);
}

TEST(Q8GEMM_8x4c1x4__SSE2, packedA_k_lt_8_bzp0) {
  TEST_REQUIRES_X86_SSE2;
  GemmBlockSparseMicrokernelTester()
      .mr(8)
      .nr(4)
      .m(8)
      .n(4)
      .k(5)
      .bZeroPoint(0)
      .test_packed(
          pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
          pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA__sse2);
}

TEST(Q8GEMM_8x4c1x4__SSE2, packedA_k_lt_8_nozp) {
  TEST_REQUIRES_X86_SSE2;
  GemmBlockSparseMicrokernelTester()
      .mr(8)
      .nr(4)
      .m(8)
      .n(4)
      .k(5)
      .aZeroPoint(0)
      .bZeroPoint(0)
      .test_packed(
          pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
          pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA__sse2);
}

TEST(Q8GEMM_8x4c1x4__SSE2, packedA_k_eq_8) {
  TEST_REQUIRES_X86_SSE2;
  GemmBlockSparseMicrokernelTester().mr(8).nr(4).m(8).n(4).k(8).test_packed(
      pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA__sse2);
}

TEST(Q8GEMM_8x4c1x4__SSE2, packedA_k_eq_8_strided_a) {
  TEST_REQUIRES_X86_SSE2;
  GemmBlockSparseMicrokernelTester()
      .mr(8)
      .nr(4)
      .m(8)
      .n(4)
      .k(8)
      .aStride(37)
      .test_packed(
          pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
          pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA__sse2);
}

TEST(Q8GEMM_8x4c1x4__SSE2, packedA_k_eq_8_strided_c) {
  TEST_REQUIRES_X86_SSE2;
  GemmBlockSparseMicrokernelTester()
      .mr(8)
      .nr(4)
      .m(8)
      .n(4)
      .k(8)
      .cStride(17)
      .test_packed(
          pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
          pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA__sse2);
}

TEST(Q8GEMM_8x4c1x4__SSE2, packedA_k_eq_8_qmin128) {
  TEST_REQUIRES_X86_SSE2;
  GemmBlockSparseMicrokernelTester().mr(8).nr(4).m(8).n(4).k(8).qmin(128).test_packed(
      pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA__sse2);
}

TEST(Q8GEMM_8x4c1x4__SSE2, packedA_k_eq_8_qmax128) {
  TEST_REQUIRES_X86_SSE2;
  GemmBlockSparseMicrokernelTester().mr(8).nr(4).m(8).n(4).k(8).qmax(128).test_packed(
      pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA__sse2);
}

TEST(Q8GEMM_8x4c1x4__SSE2, packedA_k_eq_8_azp0) {
  TEST_REQUIRES_X86_SSE2;
  GemmBlockSparseMicrokernelTester()
      .mr(8)
      .nr(4)
      .m(8)
      .n(4)
      .k(8)
      .aZeroPoint(0)
      .test_packed(
          pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
          pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA__sse2);
}

TEST(Q8GEMM_8x4c1x4__SSE2, packedA_k_eq_8_bzp0) {
  TEST_REQUIRES_X86_SSE2;
  GemmBlockSparseMicrokernelTester()
      .mr(8)
      .nr(4)
      .m(8)
      .n(4)
      .k(8)
      .bZeroPoint(0)
      .test_packed(
          pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
          pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA__sse2);
}

TEST(Q8GEMM_8x4c1x4__SSE2, packedA_k_eq_8_nozp) {
  TEST_REQUIRES_X86_SSE2;
  GemmBlockSparseMicrokernelTester()
      .mr(8)
      .nr(4)
      .m(8)
      .n(4)
      .k(8)
      .aZeroPoint(0)
      .bZeroPoint(0)
      .test_packed(
          pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
          pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA__sse2);
}

TEST(Q8GEMM_8x4c1x4__SSE2, packedA_k_gt_8) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t k = 9; k < 16; k++) {
    GemmBlockSparseMicrokernelTester().mr(8).nr(4).m(8).n(4).k(k).test_packed(
        pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
        pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA__sse2);
  }
}

TEST(Q8GEMM_8x4c1x4__SSE2, packedA_k_gt_8_strided_a) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t k = 9; k < 16; k++) {
    GemmBlockSparseMicrokernelTester()
        .mr(8)
        .nr(4)
        .m(8)
        .n(4)
        .k(k)
        .aStride(37)
        .test_packed(
            pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
            pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA__sse2);
  }
}

TEST(Q8GEMM_8x4c1x4__SSE2, packedA_k_gt_8_strided_c) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t k = 9; k < 16; k++) {
    GemmBlockSparseMicrokernelTester()
        .mr(8)
        .nr(4)
        .m(8)
        .n(4)
        .k(k)
        .cStride(17)
        .test_packed(
            pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
            pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA__sse2);
  }
}

TEST(Q8GEMM_8x4c1x4__SSE2, packedA_k_gt_8_azp0) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t k = 9; k < 16; k++) {
    GemmBlockSparseMicrokernelTester()
        .mr(8)
        .nr(4)
        .m(8)
        .n(4)
        .k(k)
        .aZeroPoint(0)
        .test_packed(
            pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
            pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA__sse2);
  }
}

TEST(Q8GEMM_8x4c1x4__SSE2, packedA_k_gt_8_bzp0) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t k = 9; k < 16; k++) {
    GemmBlockSparseMicrokernelTester()
        .mr(8)
        .nr(4)
        .m(8)
        .n(4)
        .k(k)
        .bZeroPoint(0)
        .test_packed(
            pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
            pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA__sse2);
  }
}

TEST(Q8GEMM_8x4c1x4__SSE2, packedA_k_gt_8_nozp) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t k = 9; k < 16; k++) {
    GemmBlockSparseMicrokernelTester()
        .mr(8)
        .nr(4)
        .m(8)
        .n(4)
        .k(k)
        .aZeroPoint(0)
        .bZeroPoint(0)
        .test_packed(
            pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
            pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA__sse2);
  }
}

TEST(Q8GEMM_8x4c1x4__SSE2, packedA_k_gt_8_subtile) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t k = 9; k < 16; k++) {
    for (uint32_t m = 1; m <= 8; m++) {
      for (uint32_t n = 1; n <= 4; n++) {
        GemmBlockSparseMicrokernelTester()
            .mr(8)
            .nr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(3)
            .test_packed(
                pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
                pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA__sse2);
      }
    }
  }
}

TEST(Q8GEMM_8x4c1x4__SSE2, packedA_k_div_8) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t k = 16; k < 128; k += 8) {
    GemmBlockSparseMicrokernelTester().mr(8).nr(4).m(8).n(4).k(k).test_packed(
        pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
        pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA__sse2);
  }
}

TEST(Q8GEMM_8x4c1x4__SSE2, packedA_k_div_8_strided_a) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t k = 16; k < 128; k += 8) {
    GemmBlockSparseMicrokernelTester()
        .mr(8)
        .nr(4)
        .m(8)
        .n(4)
        .k(k)
        .aStride(171)
        .test_packed(
            pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
            pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA__sse2);
  }
}

TEST(Q8GEMM_8x4c1x4__SSE2, packedA_k_div_8_strided_c) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t k = 16; k < 128; k += 8) {
    GemmBlockSparseMicrokernelTester()
        .mr(8)
        .nr(4)
        .m(8)
        .n(4)
        .k(k)
        .cStride(17)
        .test_packed(
            pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
            pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA__sse2);
  }
}

TEST(Q8GEMM_8x4c1x4__SSE2, packedA_k_div_8_subtile) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t k = 16; k < 128; k += 24) {
    for (uint32_t m = 1; m <= 8; m++) {
      for (uint32_t n = 1; n <= 4; n++) {
        GemmBlockSparseMicrokernelTester()
            .mr(8)
            .nr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(3)
            .test_packed(
                pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2,
                pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA__sse2);
      }
    }
  }
}

#endif
