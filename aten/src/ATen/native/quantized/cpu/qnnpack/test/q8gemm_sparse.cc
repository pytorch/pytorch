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

#if CPUINFO_ARCH_ARM
TEST(Q8GEMM_8x4c1x4__AARCH32_NEON, k_lt_4) {
  TEST_REQUIRES_ARM_NEON;
  GemmBlockSparseMicrokernelTester().mr(8).nr(4).m(8).n(4).k(3).test(
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4__aarch32_neon);
}

TEST(Q8GEMM_8x4c1x4__AARCH32_NEON, k_lt_4_strided_a) {
  TEST_REQUIRES_ARM_NEON;
  GemmBlockSparseMicrokernelTester()
      .mr(8)
      .nr(4)
      .m(8)
      .n(4)
      .k(3)
      .aStride(37)
      .test(pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4__aarch32_neon);
}

TEST(Q8GEMM_8x4c1x4__AARCH32_NEON, k_lt_4_strided_c) {
  TEST_REQUIRES_ARM_NEON;
  GemmBlockSparseMicrokernelTester()
      .mr(8)
      .nr(4)
      .m(8)
      .n(4)
      .k(3)
      .cStride(17)
      .test(pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4__aarch32_neon);
}

TEST(Q8GEMM_8x4c1x4__AARCH32_NEON, k_lt_4_qmin128) {
  TEST_REQUIRES_ARM_NEON;
  GemmBlockSparseMicrokernelTester().mr(8).nr(4).m(8).n(4).k(3).qmin(128).test(
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4__aarch32_neon);
}

TEST(Q8GEMM_8x4c1x4__AARCH32_NEON, k_lt_4_qmax128) {
  TEST_REQUIRES_ARM_NEON;
  GemmBlockSparseMicrokernelTester().mr(8).nr(4).m(8).n(4).k(3).qmax(128).test(
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4__aarch32_neon);
}

TEST(Q8GEMM_8x4c1x4__AARCH32_NEON, k_lt_4_azp0) {
  TEST_REQUIRES_ARM_NEON;
  GemmBlockSparseMicrokernelTester()
      .mr(8)
      .nr(4)
      .m(8)
      .n(4)
      .k(3)
      .aZeroPoint(0)
      .test(pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4__aarch32_neon);
}

TEST(Q8GEMM_8x4c1x4__AARCH32_NEON, k_lt_4_bzp0) {
  TEST_REQUIRES_ARM_NEON;
  GemmBlockSparseMicrokernelTester()
      .mr(8)
      .nr(4)
      .m(8)
      .n(4)
      .k(3)
      .bZeroPoint(0)
      .test(pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4__aarch32_neon);
}

TEST(Q8GEMM_8x4c1x4__AARCH32_NEON, k_lt_4_nozp) {
  TEST_REQUIRES_ARM_NEON;
  GemmBlockSparseMicrokernelTester()
      .mr(8)
      .nr(4)
      .m(8)
      .n(4)
      .k(3)
      .aZeroPoint(0)
      .bZeroPoint(0)
      .test(pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4__aarch32_neon);
}

TEST(Q8GEMM_8x4c1x4__AARCH32_NEON, k_lt_8) {
  TEST_REQUIRES_ARM_NEON;
  GemmBlockSparseMicrokernelTester().mr(8).nr(4).m(8).n(4).k(5).test(
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4__aarch32_neon);
}

TEST(Q8GEMM_8x4c1x4__AARCH32_NEON, k_lt_8_strided_a) {
  TEST_REQUIRES_ARM_NEON;
  GemmBlockSparseMicrokernelTester()
      .mr(8)
      .nr(4)
      .m(8)
      .n(4)
      .k(5)
      .aStride(37)
      .test(pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4__aarch32_neon);
}

TEST(Q8GEMM_8x4c1x4__AARCH32_NEON, k_lt_8_strided_c) {
  TEST_REQUIRES_ARM_NEON;
  GemmBlockSparseMicrokernelTester()
      .mr(8)
      .nr(4)
      .m(8)
      .n(4)
      .k(5)
      .cStride(17)
      .test(pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4__aarch32_neon);
}

TEST(Q8GEMM_8x4c1x4__AARCH32_NEON, k_lt_8_qmin128) {
  TEST_REQUIRES_ARM_NEON;
  GemmBlockSparseMicrokernelTester().mr(8).nr(4).m(8).n(4).k(5).qmin(128).test(
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4__aarch32_neon);
}

TEST(Q8GEMM_8x4c1x4__AARCH32_NEON, k_lt_8_qmax128) {
  TEST_REQUIRES_ARM_NEON;
  GemmBlockSparseMicrokernelTester().mr(8).nr(4).m(8).n(4).k(5).qmax(128).test(
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4__aarch32_neon);
}

TEST(Q8GEMM_8x4c1x4__AARCH32_NEON, k_lt_8_azp0) {
  TEST_REQUIRES_ARM_NEON;
  GemmBlockSparseMicrokernelTester()
      .mr(8)
      .nr(4)
      .m(8)
      .n(4)
      .k(5)
      .aZeroPoint(0)
      .test(pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4__aarch32_neon);
}

TEST(Q8GEMM_8x4c1x4__AARCH32_NEON, k_lt_8_bzp0) {
  TEST_REQUIRES_ARM_NEON;
  GemmBlockSparseMicrokernelTester()
      .mr(8)
      .nr(4)
      .m(8)
      .n(4)
      .k(5)
      .bZeroPoint(0)
      .test(pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4__aarch32_neon);
}

TEST(Q8GEMM_8x4c1x4__AARCH32_NEON, k_lt_8_nozp) {
  TEST_REQUIRES_ARM_NEON;
  GemmBlockSparseMicrokernelTester()
      .mr(8)
      .nr(4)
      .m(8)
      .n(4)
      .k(5)
      .aZeroPoint(0)
      .bZeroPoint(0)
      .test(pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4__aarch32_neon);
}

TEST(Q8GEMM_8x4c1x4__AARCH32_NEON, k_eq_8) {
  TEST_REQUIRES_ARM_NEON;
  GemmBlockSparseMicrokernelTester().mr(8).nr(4).m(8).n(4).k(8).test(
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4__aarch32_neon);
}

TEST(Q8GEMM_8x4c1x4__AARCH32_NEON, k_eq_8_strided_a) {
  TEST_REQUIRES_ARM_NEON;
  GemmBlockSparseMicrokernelTester()
      .mr(8)
      .nr(4)
      .m(8)
      .n(4)
      .k(8)
      .aStride(37)
      .test(pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4__aarch32_neon);
}

TEST(Q8GEMM_8x4c1x4__AARCH32_NEON, k_eq_8_strided_c) {
  TEST_REQUIRES_ARM_NEON;
  GemmBlockSparseMicrokernelTester()
      .mr(8)
      .nr(4)
      .m(8)
      .n(4)
      .k(8)
      .cStride(17)
      .test(pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4__aarch32_neon);
}

TEST(Q8GEMM_8x4c1x4__AARCH32_NEON, k_eq_8_qmin128) {
  TEST_REQUIRES_ARM_NEON;
  GemmBlockSparseMicrokernelTester().mr(8).nr(4).m(8).n(4).k(8).qmin(128).test(
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4__aarch32_neon);
}

TEST(Q8GEMM_8x4c1x4__AARCH32_NEON, k_eq_8_qmax128) {
  TEST_REQUIRES_ARM_NEON;
  GemmBlockSparseMicrokernelTester().mr(8).nr(4).m(8).n(4).k(8).qmax(128).test(
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4__aarch32_neon);
}

TEST(Q8GEMM_8x4c1x4__AARCH32_NEON, k_eq_8_azp0) {
  TEST_REQUIRES_ARM_NEON;
  GemmBlockSparseMicrokernelTester()
      .mr(8)
      .nr(4)
      .m(8)
      .n(4)
      .k(8)
      .aZeroPoint(0)
      .test(pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4__aarch32_neon);
}

TEST(Q8GEMM_8x4c1x4__AARCH32_NEON, k_eq_8_bzp0) {
  TEST_REQUIRES_ARM_NEON;
  GemmBlockSparseMicrokernelTester()
      .mr(8)
      .nr(4)
      .m(8)
      .n(4)
      .k(8)
      .bZeroPoint(0)
      .test(pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4__aarch32_neon);
}

TEST(Q8GEMM_8x4c1x4__AARCH32_NEON, k_eq_8_nozp) {
  TEST_REQUIRES_ARM_NEON;
  GemmBlockSparseMicrokernelTester()
      .mr(8)
      .nr(4)
      .m(8)
      .n(4)
      .k(8)
      .aZeroPoint(0)
      .bZeroPoint(0)
      .test(pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4__aarch32_neon);
}

TEST(Q8GEMM_8x4c1x4__AARCH32_NEON, k_gt_8) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    GemmBlockSparseMicrokernelTester().mr(8).nr(4).m(8).n(4).k(k).test(
        pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4__aarch32_neon);
  }
}

TEST(Q8GEMM_8x4c1x4__AARCH32_NEON, k_gt_8_strided_a) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    GemmBlockSparseMicrokernelTester()
        .mr(8)
        .nr(4)
        .m(8)
        .n(4)
        .k(k)
        .aStride(37)
        .test(pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4__aarch32_neon);
  }
}

TEST(Q8GEMM_8x4c1x4__AARCH32_NEON, k_gt_8_strided_c) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    GemmBlockSparseMicrokernelTester()
        .mr(8)
        .nr(4)
        .m(8)
        .n(4)
        .k(k)
        .cStride(17)
        .test(pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4__aarch32_neon);
  }
}

TEST(Q8GEMM_8x4c1x4__AARCH32_NEON, k_gt_8_azp0) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    GemmBlockSparseMicrokernelTester()
        .mr(8)
        .nr(4)
        .m(8)
        .n(4)
        .k(k)
        .aZeroPoint(0)
        .test(pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4__aarch32_neon);
  }
}

TEST(Q8GEMM_8x4c1x4__AARCH32_NEON, k_gt_8_bzp0) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    GemmBlockSparseMicrokernelTester()
        .mr(8)
        .nr(4)
        .m(8)
        .n(4)
        .k(k)
        .bZeroPoint(0)
        .test(pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4__aarch32_neon);
  }
}

TEST(Q8GEMM_8x4c1x4__AARCH32_NEON, k_gt_8_nozp) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    GemmBlockSparseMicrokernelTester()
        .mr(8)
        .nr(4)
        .m(8)
        .n(4)
        .k(k)
        .aZeroPoint(0)
        .bZeroPoint(0)
        .test(pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4__aarch32_neon);
  }
}

TEST(Q8GEMM_8x4c1x4__AARCH32_NEON, k_gt_8_subtile) {
  TEST_REQUIRES_ARM_NEON;
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
            .test(pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4__aarch32_neon);
      }
    }
  }
}

TEST(Q8GEMM_8x4c1x4__AARCH32_NEON, k_div_8) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 16; k < 128; k += 8) {
    GemmBlockSparseMicrokernelTester().mr(8).nr(4).m(8).n(4).k(k).test(
        pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4__aarch32_neon);
  }
}

TEST(Q8GEMM_8x4c1x4__AARCH32_NEON, k_div_8_strided_a) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 16; k < 128; k += 8) {
    GemmBlockSparseMicrokernelTester()
        .mr(8)
        .nr(4)
        .m(8)
        .n(4)
        .k(k)
        .aStride(171)
        .test(pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4__aarch32_neon);
  }
}

TEST(Q8GEMM_8x4c1x4__AARCH32_NEON, k_div_8_strided_c) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 16; k < 128; k += 8) {
    GemmBlockSparseMicrokernelTester()
        .mr(8)
        .nr(4)
        .m(8)
        .n(4)
        .k(k)
        .cStride(17)
        .test(pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4__aarch32_neon);
  }
}

TEST(Q8GEMM_8x4c1x4__AARCH32_NEON, k_div_8_subtile) {
  TEST_REQUIRES_ARM_NEON;
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
            .test(pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4__aarch32_neon);
      }
    }
  }
}

TEST(Q8GEMM_8x4c1x4__AARCH32_NEON, packedA_k_lt_4) {
  TEST_REQUIRES_ARM_NEON;
  GemmBlockSparseMicrokernelTester().mr(8).nr(4).m(8).n(4).k(3).test_packed(
      pytorch_q8gemm_sparse_packA_ukernel_8x4__aarch32_neon,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA__aarch32_neon);
}

TEST(Q8GEMM_8x4c1x4__AARCH32_NEON, packedA_k_lt_4_strided_a) {
  TEST_REQUIRES_ARM_NEON;
  GemmBlockSparseMicrokernelTester()
      .mr(8)
      .nr(4)
      .m(8)
      .n(4)
      .k(3)
      .aStride(37)
      .test_packed(
          pytorch_q8gemm_sparse_packA_ukernel_8x4__aarch32_neon,
          pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA__aarch32_neon);
}

TEST(Q8GEMM_8x4c1x4__AARCH32_NEON, packedA_k_lt_4_strided_c) {
  TEST_REQUIRES_ARM_NEON;
  GemmBlockSparseMicrokernelTester()
      .mr(8)
      .nr(4)
      .m(8)
      .n(4)
      .k(3)
      .cStride(17)
      .test_packed(
          pytorch_q8gemm_sparse_packA_ukernel_8x4__aarch32_neon,
          pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA__aarch32_neon);
}

TEST(Q8GEMM_8x4c1x4__AARCH32_NEON, packedA_k_lt_4_qmin128) {
  TEST_REQUIRES_ARM_NEON;
  GemmBlockSparseMicrokernelTester().mr(8).nr(4).m(8).n(4).k(3).qmin(128).test_packed(
      pytorch_q8gemm_sparse_packA_ukernel_8x4__aarch32_neon,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA__aarch32_neon);
}

TEST(Q8GEMM_8x4c1x4__AARCH32_NEON, packedA_k_lt_4_qmax128) {
  TEST_REQUIRES_ARM_NEON;
  GemmBlockSparseMicrokernelTester().mr(8).nr(4).m(8).n(4).k(3).qmax(128).test_packed(
      pytorch_q8gemm_sparse_packA_ukernel_8x4__aarch32_neon,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA__aarch32_neon);
}

TEST(Q8GEMM_8x4c1x4__AARCH32_NEON, packedA_k_lt_4_azp0) {
  TEST_REQUIRES_ARM_NEON;
  GemmBlockSparseMicrokernelTester()
      .mr(8)
      .nr(4)
      .m(8)
      .n(4)
      .k(3)
      .aZeroPoint(0)
      .test_packed(
          pytorch_q8gemm_sparse_packA_ukernel_8x4__aarch32_neon,
          pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA__aarch32_neon);
}

TEST(Q8GEMM_8x4c1x4__AARCH32_NEON, packedA_k_lt_4_bzp0) {
  TEST_REQUIRES_ARM_NEON;
  GemmBlockSparseMicrokernelTester()
      .mr(8)
      .nr(4)
      .m(8)
      .n(4)
      .k(3)
      .bZeroPoint(0)
      .test_packed(
          pytorch_q8gemm_sparse_packA_ukernel_8x4__aarch32_neon,
          pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA__aarch32_neon);
}

TEST(Q8GEMM_8x4c1x4__AARCH32_NEON, packedA_k_lt_4_nozp) {
  TEST_REQUIRES_ARM_NEON;
  GemmBlockSparseMicrokernelTester()
      .mr(8)
      .nr(4)
      .m(8)
      .n(4)
      .k(3)
      .aZeroPoint(0)
      .bZeroPoint(0)
      .test_packed(
          pytorch_q8gemm_sparse_packA_ukernel_8x4__aarch32_neon,
          pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA__aarch32_neon);
}

TEST(Q8GEMM_8x4c1x4__AARCH32_NEON, packedA_k_lt_8) {
  TEST_REQUIRES_ARM_NEON;
  GemmBlockSparseMicrokernelTester().mr(8).nr(4).m(8).n(4).k(5).test_packed(
      pytorch_q8gemm_sparse_packA_ukernel_8x4__aarch32_neon,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA__aarch32_neon);
}

TEST(Q8GEMM_8x4c1x4__AARCH32_NEON, packedA_k_lt_8_strided_a) {
  TEST_REQUIRES_ARM_NEON;
  GemmBlockSparseMicrokernelTester()
      .mr(8)
      .nr(4)
      .m(8)
      .n(4)
      .k(5)
      .aStride(37)
      .test_packed(
          pytorch_q8gemm_sparse_packA_ukernel_8x4__aarch32_neon,
          pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA__aarch32_neon);
}

TEST(Q8GEMM_8x4c1x4__AARCH32_NEON, packedA_k_lt_8_strided_c) {
  TEST_REQUIRES_ARM_NEON;
  GemmBlockSparseMicrokernelTester()
      .mr(8)
      .nr(4)
      .m(8)
      .n(4)
      .k(5)
      .cStride(17)
      .test_packed(
          pytorch_q8gemm_sparse_packA_ukernel_8x4__aarch32_neon,
          pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA__aarch32_neon);
}

TEST(Q8GEMM_8x4c1x4__AARCH32_NEON, packedA_k_lt_8_qmin128) {
  TEST_REQUIRES_ARM_NEON;
  GemmBlockSparseMicrokernelTester().mr(8).nr(4).m(8).n(4).k(5).qmin(128).test_packed(
      pytorch_q8gemm_sparse_packA_ukernel_8x4__aarch32_neon,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA__aarch32_neon);
}

TEST(Q8GEMM_8x4c1x4__AARCH32_NEON, packedA_k_lt_8_qmax128) {
  TEST_REQUIRES_ARM_NEON;
  GemmBlockSparseMicrokernelTester().mr(8).nr(4).m(8).n(4).k(5).qmax(128).test_packed(
      pytorch_q8gemm_sparse_packA_ukernel_8x4__aarch32_neon,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA__aarch32_neon);
}

TEST(Q8GEMM_8x4c1x4__AARCH32_NEON, packedA_k_lt_8_azp0) {
  TEST_REQUIRES_ARM_NEON;
  GemmBlockSparseMicrokernelTester()
      .mr(8)
      .nr(4)
      .m(8)
      .n(4)
      .k(5)
      .aZeroPoint(0)
      .test_packed(
          pytorch_q8gemm_sparse_packA_ukernel_8x4__aarch32_neon,
          pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA__aarch32_neon);
}

TEST(Q8GEMM_8x4c1x4__AARCH32_NEON, packedA_k_lt_8_bzp0) {
  TEST_REQUIRES_ARM_NEON;
  GemmBlockSparseMicrokernelTester()
      .mr(8)
      .nr(4)
      .m(8)
      .n(4)
      .k(5)
      .bZeroPoint(0)
      .test_packed(
          pytorch_q8gemm_sparse_packA_ukernel_8x4__aarch32_neon,
          pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA__aarch32_neon);
}

TEST(Q8GEMM_8x4c1x4__AARCH32_NEON, packedA_k_lt_8_nozp) {
  TEST_REQUIRES_ARM_NEON;
  GemmBlockSparseMicrokernelTester()
      .mr(8)
      .nr(4)
      .m(8)
      .n(4)
      .k(5)
      .aZeroPoint(0)
      .bZeroPoint(0)
      .test_packed(
          pytorch_q8gemm_sparse_packA_ukernel_8x4__aarch32_neon,
          pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA__aarch32_neon);
}

TEST(Q8GEMM_8x4c1x4__AARCH32_NEON, packedA_k_eq_8) {
  TEST_REQUIRES_ARM_NEON;
  GemmBlockSparseMicrokernelTester().mr(8).nr(4).m(8).n(4).k(8).test_packed(
      pytorch_q8gemm_sparse_packA_ukernel_8x4__aarch32_neon,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA__aarch32_neon);
}

TEST(Q8GEMM_8x4c1x4__AARCH32_NEON, packedA_k_eq_8_strided_a) {
  TEST_REQUIRES_ARM_NEON;
  GemmBlockSparseMicrokernelTester()
      .mr(8)
      .nr(4)
      .m(8)
      .n(4)
      .k(8)
      .aStride(37)
      .test_packed(
          pytorch_q8gemm_sparse_packA_ukernel_8x4__aarch32_neon,
          pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA__aarch32_neon);
}

TEST(Q8GEMM_8x4c1x4__AARCH32_NEON, packedA_k_eq_8_strided_c) {
  TEST_REQUIRES_ARM_NEON;
  GemmBlockSparseMicrokernelTester()
      .mr(8)
      .nr(4)
      .m(8)
      .n(4)
      .k(8)
      .cStride(17)
      .test_packed(
          pytorch_q8gemm_sparse_packA_ukernel_8x4__aarch32_neon,
          pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA__aarch32_neon);
}

TEST(Q8GEMM_8x4c1x4__AARCH32_NEON, packedA_k_eq_8_qmin128) {
  TEST_REQUIRES_ARM_NEON;
  GemmBlockSparseMicrokernelTester().mr(8).nr(4).m(8).n(4).k(8).qmin(128).test_packed(
      pytorch_q8gemm_sparse_packA_ukernel_8x4__aarch32_neon,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA__aarch32_neon);
}

TEST(Q8GEMM_8x4c1x4__AARCH32_NEON, packedA_k_eq_8_qmax128) {
  TEST_REQUIRES_ARM_NEON;
  GemmBlockSparseMicrokernelTester().mr(8).nr(4).m(8).n(4).k(8).qmax(128).test_packed(
      pytorch_q8gemm_sparse_packA_ukernel_8x4__aarch32_neon,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA__aarch32_neon);
}

TEST(Q8GEMM_8x4c1x4__AARCH32_NEON, packedA_k_eq_8_azp0) {
  TEST_REQUIRES_ARM_NEON;
  GemmBlockSparseMicrokernelTester()
      .mr(8)
      .nr(4)
      .m(8)
      .n(4)
      .k(8)
      .aZeroPoint(0)
      .test_packed(
          pytorch_q8gemm_sparse_packA_ukernel_8x4__aarch32_neon,
          pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA__aarch32_neon);
}

TEST(Q8GEMM_8x4c1x4__AARCH32_NEON, packedA_k_eq_8_bzp0) {
  TEST_REQUIRES_ARM_NEON;
  GemmBlockSparseMicrokernelTester()
      .mr(8)
      .nr(4)
      .m(8)
      .n(4)
      .k(8)
      .bZeroPoint(0)
      .test_packed(
          pytorch_q8gemm_sparse_packA_ukernel_8x4__aarch32_neon,
          pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA__aarch32_neon);
}

TEST(Q8GEMM_8x4c1x4__AARCH32_NEON, packedA_k_eq_8_nozp) {
  TEST_REQUIRES_ARM_NEON;
  GemmBlockSparseMicrokernelTester()
      .mr(8)
      .nr(4)
      .m(8)
      .n(4)
      .k(8)
      .aZeroPoint(0)
      .bZeroPoint(0)
      .test_packed(
          pytorch_q8gemm_sparse_packA_ukernel_8x4__aarch32_neon,
          pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA__aarch32_neon);
}

TEST(Q8GEMM_8x4c1x4__AARCH32_NEON, packedA_k_gt_8) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    GemmBlockSparseMicrokernelTester().mr(8).nr(4).m(8).n(4).k(k).test_packed(
        pytorch_q8gemm_sparse_packA_ukernel_8x4__aarch32_neon,
        pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA__aarch32_neon);
  }
}

TEST(Q8GEMM_8x4c1x4__AARCH32_NEON, packedA_k_gt_8_strided_a) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    GemmBlockSparseMicrokernelTester()
        .mr(8)
        .nr(4)
        .m(8)
        .n(4)
        .k(k)
        .aStride(37)
        .test_packed(
            pytorch_q8gemm_sparse_packA_ukernel_8x4__aarch32_neon,
            pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA__aarch32_neon);
  }
}

TEST(Q8GEMM_8x4c1x4__AARCH32_NEON, packedA_k_gt_8_strided_c) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    GemmBlockSparseMicrokernelTester()
        .mr(8)
        .nr(4)
        .m(8)
        .n(4)
        .k(k)
        .cStride(17)
        .test_packed(
            pytorch_q8gemm_sparse_packA_ukernel_8x4__aarch32_neon,
            pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA__aarch32_neon);
  }
}

TEST(Q8GEMM_8x4c1x4__AARCH32_NEON, packedA_k_gt_8_azp0) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    GemmBlockSparseMicrokernelTester()
        .mr(8)
        .nr(4)
        .m(8)
        .n(4)
        .k(k)
        .aZeroPoint(0)
        .test_packed(
            pytorch_q8gemm_sparse_packA_ukernel_8x4__aarch32_neon,
            pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA__aarch32_neon);
  }
}

TEST(Q8GEMM_8x4c1x4__AARCH32_NEON, packedA_k_gt_8_bzp0) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    GemmBlockSparseMicrokernelTester()
        .mr(8)
        .nr(4)
        .m(8)
        .n(4)
        .k(k)
        .bZeroPoint(0)
        .test_packed(
            pytorch_q8gemm_sparse_packA_ukernel_8x4__aarch32_neon,
            pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA__aarch32_neon);
  }
}

TEST(Q8GEMM_8x4c1x4__AARCH32_NEON, packedA_k_gt_8_nozp) {
  TEST_REQUIRES_ARM_NEON;
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
            pytorch_q8gemm_sparse_packA_ukernel_8x4__aarch32_neon,
            pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA__aarch32_neon);
  }
}

TEST(Q8GEMM_8x4c1x4__AARCH32_NEON, packedA_k_gt_8_subtile) {
  TEST_REQUIRES_ARM_NEON;
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
                pytorch_q8gemm_sparse_packA_ukernel_8x4__aarch32_neon,
                pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA__aarch32_neon);
      }
    }
  }
}

TEST(Q8GEMM_8x4c1x4__AARCH32_NEON, packedA_k_div_8) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 16; k < 128; k += 8) {
    GemmBlockSparseMicrokernelTester().mr(8).nr(4).m(8).n(4).k(k).test_packed(
        pytorch_q8gemm_sparse_packA_ukernel_8x4__aarch32_neon,
        pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA__aarch32_neon);
  }
}

TEST(Q8GEMM_8x4c1x4__AARCH32_NEON, packedA_k_div_8_strided_a) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 16; k < 128; k += 8) {
    GemmBlockSparseMicrokernelTester()
        .mr(8)
        .nr(4)
        .m(8)
        .n(4)
        .k(k)
        .aStride(171)
        .test_packed(
            pytorch_q8gemm_sparse_packA_ukernel_8x4__aarch32_neon,
            pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA__aarch32_neon);
  }
}

TEST(Q8GEMM_8x4c1x4__AARCH32_NEON, packedA_k_div_8_strided_c) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 16; k < 128; k += 8) {
    GemmBlockSparseMicrokernelTester()
        .mr(8)
        .nr(4)
        .m(8)
        .n(4)
        .k(k)
        .cStride(17)
        .test_packed(
            pytorch_q8gemm_sparse_packA_ukernel_8x4__aarch32_neon,
            pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA__aarch32_neon);
  }
}

TEST(Q8GEMM_8x4c1x4__AARCH32_NEON, packedA_k_div_8_subtile) {
  TEST_REQUIRES_ARM_NEON;
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
                pytorch_q8gemm_sparse_packA_ukernel_8x4__aarch32_neon,
                pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA__aarch32_neon);
      }
    }
  }
}

TEST(Q8GEMM_4x8c1x4__AARCH32_NEON, packedA_k_lt_4) {
  TEST_REQUIRES_ARM_NEON;
  GemmBlockSparseMicrokernelTester().mr(4).nr(8).m(4).n(8).k(4).test_packed(
      pytorch_q8gemm_sparse_packA_ukernel_4x4__aarch32_neon,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_4x8_packedA__aarch32_neon);
}

TEST(Q8GEMM_4x8c1x4__AARCH32_NEON, packedA_k_lt_4_strided_a) {
  TEST_REQUIRES_ARM_NEON;
  GemmBlockSparseMicrokernelTester()
      .mr(4)
      .nr(8)
      .m(4)
      .n(8)
      .k(3)
      .aStride(37)
      .test_packed(
          pytorch_q8gemm_sparse_packA_ukernel_4x4__aarch32_neon,
          pytorch_q8gemm_dq_sparse_1x4_ukernel_4x8_packedA__aarch32_neon);
}

TEST(Q8GEMM_4x8c1x4__AARCH32_NEON, packedA_k_lt_4_strided_c) {
  TEST_REQUIRES_ARM_NEON;
  GemmBlockSparseMicrokernelTester()
      .mr(4)
      .nr(8)
      .m(4)
      .n(8)
      .k(3)
      .cStride(17)
      .test_packed(
          pytorch_q8gemm_sparse_packA_ukernel_4x4__aarch32_neon,
          pytorch_q8gemm_dq_sparse_1x4_ukernel_4x8_packedA__aarch32_neon);
}

TEST(Q8GEMM_4x8c1x4__AARCH32_NEON, packedA_k_lt_4_qmin128) {
  TEST_REQUIRES_ARM_NEON;
  GemmBlockSparseMicrokernelTester().mr(4).nr(8).m(4).n(8).k(3).qmin(128).test_packed(
      pytorch_q8gemm_sparse_packA_ukernel_4x4__aarch32_neon,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_4x8_packedA__aarch32_neon);
}

TEST(Q8GEMM_4x8c1x4__AARCH32_NEON, packedA_k_lt_4_qmax128) {
  TEST_REQUIRES_ARM_NEON;
  GemmBlockSparseMicrokernelTester().mr(4).nr(8).m(4).n(8).k(3).qmax(128).test_packed(
      pytorch_q8gemm_sparse_packA_ukernel_4x4__aarch32_neon,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_4x8_packedA__aarch32_neon);
}

TEST(Q8GEMM_4x8c1x4__AARCH32_NEON, packedA_k_lt_4_azp0) {
  TEST_REQUIRES_ARM_NEON;
  GemmBlockSparseMicrokernelTester()
      .mr(4)
      .nr(8)
      .m(4)
      .n(8)
      .k(3)
      .aZeroPoint(0)
      .test_packed(
          pytorch_q8gemm_sparse_packA_ukernel_4x4__aarch32_neon,
          pytorch_q8gemm_dq_sparse_1x4_ukernel_4x8_packedA__aarch32_neon);
}

TEST(Q8GEMM_4x8c1x4__AARCH32_NEON, packedA_k_lt_4_bzp0) {
  TEST_REQUIRES_ARM_NEON;
  GemmBlockSparseMicrokernelTester()
      .mr(4)
      .nr(8)
      .m(4)
      .n(8)
      .k(3)
      .bZeroPoint(0)
      .test_packed(
          pytorch_q8gemm_sparse_packA_ukernel_4x4__aarch32_neon,
          pytorch_q8gemm_dq_sparse_1x4_ukernel_4x8_packedA__aarch32_neon);
}

TEST(Q8GEMM_4x8c1x4__AARCH32_NEON, packedA_k_lt_4_nozp) {
  TEST_REQUIRES_ARM_NEON;
  GemmBlockSparseMicrokernelTester()
      .mr(4)
      .nr(8)
      .m(4)
      .n(8)
      .k(3)
      .aZeroPoint(0)
      .bZeroPoint(0)
      .test_packed(
          pytorch_q8gemm_sparse_packA_ukernel_4x4__aarch32_neon,
          pytorch_q8gemm_dq_sparse_1x4_ukernel_4x8_packedA__aarch32_neon);
}

TEST(Q8GEMM_4x8c1x4__AARCH32_NEON, packedA_k_lt_8) {
  TEST_REQUIRES_ARM_NEON;
  GemmBlockSparseMicrokernelTester().mr(4).nr(8).m(4).n(8).k(5).test_packed(
      pytorch_q8gemm_sparse_packA_ukernel_4x4__aarch32_neon,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_4x8_packedA__aarch32_neon);
}

TEST(Q8GEMM_4x8c1x4__AARCH32_NEON, packedA_k_lt_8_strided_a) {
  TEST_REQUIRES_ARM_NEON;
  GemmBlockSparseMicrokernelTester()
      .mr(4)
      .nr(8)
      .m(4)
      .n(8)
      .k(5)
      .aStride(37)
      .test_packed(
          pytorch_q8gemm_sparse_packA_ukernel_4x4__aarch32_neon,
          pytorch_q8gemm_dq_sparse_1x4_ukernel_4x8_packedA__aarch32_neon);
}

TEST(Q8GEMM_4x8c1x4__AARCH32_NEON, packedA_k_lt_8_strided_c) {
  TEST_REQUIRES_ARM_NEON;
  GemmBlockSparseMicrokernelTester()
      .mr(4)
      .nr(8)
      .m(4)
      .n(8)
      .k(5)
      .cStride(17)
      .test_packed(
          pytorch_q8gemm_sparse_packA_ukernel_4x4__aarch32_neon,
          pytorch_q8gemm_dq_sparse_1x4_ukernel_4x8_packedA__aarch32_neon);
}

TEST(Q8GEMM_4x8c1x4__AARCH32_NEON, packedA_k_lt_8_qmin128) {
  TEST_REQUIRES_ARM_NEON;
  GemmBlockSparseMicrokernelTester().mr(4).nr(8).m(4).n(8).k(5).qmin(128).test_packed(
      pytorch_q8gemm_sparse_packA_ukernel_4x4__aarch32_neon,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_4x8_packedA__aarch32_neon);
}

TEST(Q8GEMM_4x8c1x4__AARCH32_NEON, packedA_k_lt_8_qmax128) {
  TEST_REQUIRES_ARM_NEON;
  GemmBlockSparseMicrokernelTester().mr(4).nr(8).m(4).n(8).k(5).qmax(128).test_packed(
      pytorch_q8gemm_sparse_packA_ukernel_4x4__aarch32_neon,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_4x8_packedA__aarch32_neon);
}

TEST(Q8GEMM_4x8c1x4__AARCH32_NEON, packedA_k_lt_8_azp0) {
  TEST_REQUIRES_ARM_NEON;
  GemmBlockSparseMicrokernelTester()
      .mr(4)
      .nr(8)
      .m(4)
      .n(8)
      .k(5)
      .aZeroPoint(0)
      .test_packed(
          pytorch_q8gemm_sparse_packA_ukernel_4x4__aarch32_neon,
          pytorch_q8gemm_dq_sparse_1x4_ukernel_4x8_packedA__aarch32_neon);
}

TEST(Q8GEMM_4x8c1x4__AARCH32_NEON, packedA_k_lt_8_bzp0) {
  TEST_REQUIRES_ARM_NEON;
  GemmBlockSparseMicrokernelTester()
      .mr(4)
      .nr(8)
      .m(4)
      .n(8)
      .k(5)
      .bZeroPoint(0)
      .test_packed(
          pytorch_q8gemm_sparse_packA_ukernel_4x4__aarch32_neon,
          pytorch_q8gemm_dq_sparse_1x4_ukernel_4x8_packedA__aarch32_neon);
}

TEST(Q8GEMM_4x8c1x4__AARCH32_NEON, packedA_k_lt_8_nozp) {
  TEST_REQUIRES_ARM_NEON;
  GemmBlockSparseMicrokernelTester()
      .mr(4)
      .nr(8)
      .m(4)
      .n(8)
      .k(5)
      .aZeroPoint(0)
      .bZeroPoint(0)
      .test_packed(
          pytorch_q8gemm_sparse_packA_ukernel_4x4__aarch32_neon,
          pytorch_q8gemm_dq_sparse_1x4_ukernel_4x8_packedA__aarch32_neon);
}

TEST(Q8GEMM_4x8c1x4__AARCH32_NEON, packedA_k_eq_8) {
  TEST_REQUIRES_ARM_NEON;
  GemmBlockSparseMicrokernelTester().mr(4).nr(8).m(4).n(8).k(8).test_packed(
      pytorch_q8gemm_sparse_packA_ukernel_4x4__aarch32_neon,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_4x8_packedA__aarch32_neon);
}

TEST(Q8GEMM_4x8c1x4__AARCH32_NEON, packedA_k_eq_8_strided_a) {
  TEST_REQUIRES_ARM_NEON;
  GemmBlockSparseMicrokernelTester()
      .mr(4)
      .nr(8)
      .m(4)
      .n(8)
      .k(8)
      .aStride(37)
      .test_packed(
          pytorch_q8gemm_sparse_packA_ukernel_4x4__aarch32_neon,
          pytorch_q8gemm_dq_sparse_1x4_ukernel_4x8_packedA__aarch32_neon);
}

TEST(Q8GEMM_4x8c1x4__AARCH32_NEON, packedA_k_eq_8_strided_c) {
  TEST_REQUIRES_ARM_NEON;
  GemmBlockSparseMicrokernelTester()
      .mr(4)
      .nr(8)
      .m(4)
      .n(8)
      .k(8)
      .cStride(17)
      .test_packed(
          pytorch_q8gemm_sparse_packA_ukernel_4x4__aarch32_neon,
          pytorch_q8gemm_dq_sparse_1x4_ukernel_4x8_packedA__aarch32_neon);
}

TEST(Q8GEMM_4x8c1x4__AARCH32_NEON, packedA_k_eq_8_qmin128) {
  TEST_REQUIRES_ARM_NEON;
  GemmBlockSparseMicrokernelTester().mr(4).nr(8).m(4).n(8).k(8).qmin(128).test_packed(
      pytorch_q8gemm_sparse_packA_ukernel_4x4__aarch32_neon,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_4x8_packedA__aarch32_neon);
}

TEST(Q8GEMM_4x8c1x4__AARCH32_NEON, packedA_k_eq_8_qmax128) {
  TEST_REQUIRES_ARM_NEON;
  GemmBlockSparseMicrokernelTester().mr(4).nr(8).m(4).n(8).k(8).qmax(128).test_packed(
      pytorch_q8gemm_sparse_packA_ukernel_4x4__aarch32_neon,
      pytorch_q8gemm_dq_sparse_1x4_ukernel_4x8_packedA__aarch32_neon);
}

TEST(Q8GEMM_4x8c1x4__AARCH32_NEON, packedA_k_eq_8_azp0) {
  TEST_REQUIRES_ARM_NEON;
  GemmBlockSparseMicrokernelTester()
      .mr(4)
      .nr(8)
      .m(4)
      .n(8)
      .k(8)
      .aZeroPoint(0)
      .test_packed(
          pytorch_q8gemm_sparse_packA_ukernel_4x4__aarch32_neon,
          pytorch_q8gemm_dq_sparse_1x4_ukernel_4x8_packedA__aarch32_neon);
}

TEST(Q8GEMM_4x8c1x4__AARCH32_NEON, packedA_k_eq_8_bzp0) {
  TEST_REQUIRES_ARM_NEON;
  GemmBlockSparseMicrokernelTester()
      .mr(4)
      .nr(8)
      .m(4)
      .n(8)
      .k(8)
      .bZeroPoint(0)
      .test_packed(
          pytorch_q8gemm_sparse_packA_ukernel_4x4__aarch32_neon,
          pytorch_q8gemm_dq_sparse_1x4_ukernel_4x8_packedA__aarch32_neon);
}

TEST(Q8GEMM_4x8c1x4__AARCH32_NEON, packedA_k_eq_8_nozp) {
  TEST_REQUIRES_ARM_NEON;
  GemmBlockSparseMicrokernelTester()
      .mr(4)
      .nr(8)
      .m(4)
      .n(8)
      .k(8)
      .aZeroPoint(0)
      .bZeroPoint(0)
      .test_packed(
          pytorch_q8gemm_sparse_packA_ukernel_4x4__aarch32_neon,
          pytorch_q8gemm_dq_sparse_1x4_ukernel_4x8_packedA__aarch32_neon);
}

TEST(Q8GEMM_4x8c1x4__AARCH32_NEON, packedA_k_gt_8) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    GemmBlockSparseMicrokernelTester().mr(4).nr(8).m(4).n(8).k(k).test_packed(
        pytorch_q8gemm_sparse_packA_ukernel_4x4__aarch32_neon,
        pytorch_q8gemm_dq_sparse_1x4_ukernel_4x8_packedA__aarch32_neon);
  }
}

TEST(Q8GEMM_4x8c1x4__AARCH32_NEON, packedA_k_gt_8_strided_a) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    GemmBlockSparseMicrokernelTester()
        .mr(4)
        .nr(8)
        .m(4)
        .n(8)
        .k(k)
        .aStride(37)
        .test_packed(
            pytorch_q8gemm_sparse_packA_ukernel_4x4__aarch32_neon,
            pytorch_q8gemm_dq_sparse_1x4_ukernel_4x8_packedA__aarch32_neon);
  }
}

TEST(Q8GEMM_4x8c1x4__AARCH32_NEON, packedA_k_gt_8_strided_c) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    GemmBlockSparseMicrokernelTester()
        .mr(4)
        .nr(8)
        .m(4)
        .n(8)
        .k(k)
        .cStride(17)
        .test_packed(
            pytorch_q8gemm_sparse_packA_ukernel_4x4__aarch32_neon,
            pytorch_q8gemm_dq_sparse_1x4_ukernel_4x8_packedA__aarch32_neon);
  }
}

TEST(Q8GEMM_4x8c1x4__AARCH32_NEON, packedA_k_gt_8_azp0) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    GemmBlockSparseMicrokernelTester()
        .mr(4)
        .nr(8)
        .m(4)
        .n(8)
        .k(k)
        .aZeroPoint(0)
        .test_packed(
            pytorch_q8gemm_sparse_packA_ukernel_4x4__aarch32_neon,
            pytorch_q8gemm_dq_sparse_1x4_ukernel_4x8_packedA__aarch32_neon);
  }
}

TEST(Q8GEMM_4x8c1x4__AARCH32_NEON, packedA_k_gt_8_bzp0) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    GemmBlockSparseMicrokernelTester()
        .mr(4)
        .nr(8)
        .m(4)
        .n(8)
        .k(k)
        .bZeroPoint(0)
        .test_packed(
            pytorch_q8gemm_sparse_packA_ukernel_4x4__aarch32_neon,
            pytorch_q8gemm_dq_sparse_1x4_ukernel_4x8_packedA__aarch32_neon);
  }
}

TEST(Q8GEMM_4x8c1x4__AARCH32_NEON, packedA_k_gt_8_nozp) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    GemmBlockSparseMicrokernelTester()
        .mr(4)
        .nr(8)
        .m(4)
        .n(8)
        .k(k)
        .aZeroPoint(0)
        .bZeroPoint(0)
        .test_packed(
            pytorch_q8gemm_sparse_packA_ukernel_4x4__aarch32_neon,
            pytorch_q8gemm_dq_sparse_1x4_ukernel_4x8_packedA__aarch32_neon);
  }
}

TEST(Q8GEMM_4x8c1x4__AARCH32_NEON, packedA_k_gt_8_subtile) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    for (uint32_t m = 1; m <= 4; m++) {
      for (uint32_t n = 1; n <= 8; n++) {
        GemmBlockSparseMicrokernelTester()
            .mr(4)
            .nr(8)
            .m(4)
            .n(8)
            .k(k)
            .iterations(3)
            .test_packed(
                pytorch_q8gemm_sparse_packA_ukernel_4x4__aarch32_neon,
                pytorch_q8gemm_dq_sparse_1x4_ukernel_4x8_packedA__aarch32_neon);
      }
    }
  }
}

TEST(Q8GEMM_4x8c1x4__AARCH32_NEON, packedA_k_div_8) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 16; k < 128; k += 8) {
    GemmBlockSparseMicrokernelTester().mr(4).nr(8).m(4).n(8).k(k).test_packed(
        pytorch_q8gemm_sparse_packA_ukernel_4x4__aarch32_neon,
        pytorch_q8gemm_dq_sparse_1x4_ukernel_4x8_packedA__aarch32_neon);
  }
}

TEST(Q8GEMM_4x8c1x4__AARCH32_NEON, packedA_k_div_8_strided_a) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 16; k < 128; k += 8) {
    GemmBlockSparseMicrokernelTester()
        .mr(4)
        .nr(8)
        .m(4)
        .n(8)
        .k(k)
        .aStride(171)
        .test_packed(
            pytorch_q8gemm_sparse_packA_ukernel_4x4__aarch32_neon,
            pytorch_q8gemm_dq_sparse_1x4_ukernel_4x8_packedA__aarch32_neon);
  }
}

TEST(Q8GEMM_4x8c1x4__AARCH32_NEON, packedA_k_div_8_strided_c) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 16; k < 128; k += 8) {
    GemmBlockSparseMicrokernelTester()
        .mr(4)
        .nr(8)
        .m(4)
        .n(8)
        .k(k)
        .cStride(17)
        .test_packed(
            pytorch_q8gemm_sparse_packA_ukernel_4x4__aarch32_neon,
            pytorch_q8gemm_dq_sparse_1x4_ukernel_4x8_packedA__aarch32_neon);
  }
}

TEST(Q8GEMM_4x8c1x4__AARCH32_NEON, packedA_k_div_8_subtile) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 16; k < 128; k += 24) {
    for (uint32_t m = 1; m <= 4; m++) {
      for (uint32_t n = 1; n <= 8; n++) {
        GemmBlockSparseMicrokernelTester()
            .mr(4)
            .nr(8)
            .m(m)
            .n(n)
            .k(k)
            .iterations(3)
            .test_packed(
                pytorch_q8gemm_sparse_packA_ukernel_4x4__aarch32_neon,
                pytorch_q8gemm_dq_sparse_1x4_ukernel_4x8_packedA__aarch32_neon);
      }
    }
  }
}
#endif

#if CPUINFO_ARCH_ARM64
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
