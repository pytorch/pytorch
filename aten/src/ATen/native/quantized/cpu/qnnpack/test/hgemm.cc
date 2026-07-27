/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cpuinfo.h>
#include <gtest/gtest.h>

#include <qnnpack/hgemm.h>
#include <qnnpack/isa-checks.h>

#include "gemm-microkernel-tester.h"

#if CPUINFO_ARCH_ARM
TEST(HGEMM_8x8__AARCH32_NEONFP16ARITH, k_eq_4) {
  TEST_REQUIRES_ARM_NEON_FP16_ARITH;
  GemmMicrokernelTester().mr(8).nr(8).np(8).kr(1).m(8).n(8).k(4).test(
      pytorch_hgemm_ukernel_8x8__aarch32_neonfp16arith);
}

TEST(HGEMM_8x8__AARCH32_NEONFP16ARITH, k_eq_4_strided_a) {
  TEST_REQUIRES_ARM_NEON_FP16_ARITH;
  GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .np(8)
      .kr(1)
      .m(8)
      .n(8)
      .k(4)
      .aStride(37)
      .test(pytorch_hgemm_ukernel_8x8__aarch32_neonfp16arith);
}

TEST(HGEMM_8x8__AARCH32_NEONFP16ARITH, k_eq_4_strided_c) {
  TEST_REQUIRES_ARM_NEON_FP16_ARITH;
  GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .np(8)
      .kr(1)
      .m(8)
      .n(8)
      .k(4)
      .cStride(17)
      .test(pytorch_hgemm_ukernel_8x8__aarch32_neonfp16arith);
}

TEST(HGEMM_8x8__AARCH32_NEONFP16ARITH, k_eq_4_qmin128) {
  TEST_REQUIRES_ARM_NEON_FP16_ARITH;
  GemmMicrokernelTester().mr(8).nr(8).np(8).kr(1).m(8).n(8).k(4).qmin(128).test(
      pytorch_hgemm_ukernel_8x8__aarch32_neonfp16arith);
}

TEST(HGEMM_8x8__AARCH32_NEONFP16ARITH, k_eq_4_qmax128) {
  TEST_REQUIRES_ARM_NEON_FP16_ARITH;
  GemmMicrokernelTester().mr(8).nr(8).np(8).kr(1).m(8).n(8).k(4).qmax(128).test(
      pytorch_hgemm_ukernel_8x8__aarch32_neonfp16arith);
}

TEST(HGEMM_8x8__AARCH32_NEONFP16ARITH, k_gt_4) {
  TEST_REQUIRES_ARM_NEON_FP16_ARITH;
  for (size_t k = 5; k < 8; k++) {
    GemmMicrokernelTester().mr(8).nr(8).np(8).kr(1).m(8).n(8).k(k).test(
        pytorch_hgemm_ukernel_8x8__aarch32_neonfp16arith);
  }
}

TEST(HGEMM_8x8__AARCH32_NEONFP16ARITH, k_gt_4_strided_a) {
  TEST_REQUIRES_ARM_NEON_FP16_ARITH;
  for (size_t k = 5; k < 8; k++) {
    GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .np(8)
        .kr(1)
        .m(8)
        .n(8)
        .k(k)
        .aStride(37)
        .test(pytorch_hgemm_ukernel_8x8__aarch32_neonfp16arith);
  }
}

TEST(HGEMM_8x8__AARCH32_NEONFP16ARITH, k_gt_4_strided_c) {
  TEST_REQUIRES_ARM_NEON_FP16_ARITH;
  for (size_t k = 5; k < 8; k++) {
    GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .np(8)
        .kr(1)
        .m(8)
        .n(8)
        .k(k)
        .cStride(17)
        .test(pytorch_hgemm_ukernel_8x8__aarch32_neonfp16arith);
  }
}

TEST(HGEMM_8x8__AARCH32_NEONFP16ARITH, k_gt_4_subtile) {
  TEST_REQUIRES_ARM_NEON_FP16_ARITH;
  for (size_t k = 5; k < 8; k++) {
    for (uint32_t m = 1; m <= 8; m++) {
      for (uint32_t n = 1; n <= 8; n++) {
        GemmMicrokernelTester()
            .mr(8)
            .nr(8)
            .np(8)
            .kr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(3)
            .test(pytorch_hgemm_ukernel_8x8__aarch32_neonfp16arith);
      }
    }
  }
}

TEST(HGEMM_8x8__AARCH32_NEONFP16ARITH, k_div_4) {
  TEST_REQUIRES_ARM_NEON_FP16_ARITH;
  for (size_t k = 8; k < 64; k += 4) {
    GemmMicrokernelTester().mr(8).nr(8).np(8).kr(1).m(8).n(8).k(k).test(
        pytorch_hgemm_ukernel_8x8__aarch32_neonfp16arith);
  }
}

TEST(HGEMM_8x8__AARCH32_NEONFP16ARITH, k_div_4_strided_a) {
  TEST_REQUIRES_ARM_NEON_FP16_ARITH;
  for (size_t k = 8; k < 64; k += 4) {
    GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .np(8)
        .kr(1)
        .m(8)
        .n(8)
        .k(k)
        .aStride(171)
        .test(pytorch_hgemm_ukernel_8x8__aarch32_neonfp16arith);
  }
}

TEST(HGEMM_8x8__AARCH32_NEONFP16ARITH, k_div_4_strided_c) {
  TEST_REQUIRES_ARM_NEON_FP16_ARITH;
  for (size_t k = 8; k < 64; k += 4) {
    GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .np(8)
        .kr(1)
        .m(8)
        .n(8)
        .k(k)
        .cStride(17)
        .test(pytorch_hgemm_ukernel_8x8__aarch32_neonfp16arith);
  }
}

TEST(HGEMM_8x8__AARCH32_NEONFP16ARITH, k_div_4_subtile) {
  TEST_REQUIRES_ARM_NEON_FP16_ARITH;
  for (size_t k = 8; k < 64; k += 12) {
    for (uint32_t m = 1; m <= 1; m++) {
      for (uint32_t n = 8; n <= 8; n++) {
        GemmMicrokernelTester()
            .mr(8)
            .nr(8)
            .np(8)
            .kr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(3)
            .test(pytorch_hgemm_ukernel_8x8__aarch32_neonfp16arith);
      }
    }
  }
}
#endif
