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
#include <qnnpack/q8gemm.h>

#include "gemm-microkernel-tester.h"

#if CPUINFO_ARCH_ARM
TEST(Q8GEMM_4x8__AARCH32_NEON, k_eq_8) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester().mr(4).nr(8).np(8).kr(1).m(4).n(8).k(8).test(
      pytorch_q8gemm_ukernel_4x8__aarch32_neon);
}

TEST(Q8GEMM_4x8__AARCH32_NEON, k_eq_8_strided_a) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .np(8)
      .kr(1)
      .m(4)
      .n(8)
      .k(8)
      .aStride(37)
      .test(pytorch_q8gemm_ukernel_4x8__aarch32_neon);
}

TEST(Q8GEMM_4x8__AARCH32_NEON, k_eq_8_strided_c) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .np(8)
      .kr(1)
      .m(4)
      .n(8)
      .k(8)
      .cStride(17)
      .test(pytorch_q8gemm_ukernel_4x8__aarch32_neon);
}

TEST(Q8GEMM_4x8__AARCH32_NEON, k_eq_8_qmin128) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester().mr(4).nr(8).np(8).kr(1).m(4).n(8).k(8).qmin(128).test(
      pytorch_q8gemm_ukernel_4x8__aarch32_neon);
}

TEST(Q8GEMM_4x8__AARCH32_NEON, k_eq_8_qmax128) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester().mr(4).nr(8).np(8).kr(1).m(4).n(8).k(8).qmax(128).test(
      pytorch_q8gemm_ukernel_4x8__aarch32_neon);
}

TEST(Q8GEMM_4x8__AARCH32_NEON, k_eq_8_azp0) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .np(8)
      .kr(1)
      .m(4)
      .n(8)
      .k(8)
      .aZeroPoint(0)
      .test(pytorch_q8gemm_ukernel_4x8__aarch32_neon);
}

TEST(Q8GEMM_4x8__AARCH32_NEON, k_eq_8_bzp0) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .np(8)
      .kr(1)
      .m(4)
      .n(8)
      .k(8)
      .bZeroPoint(0)
      .test(pytorch_q8gemm_ukernel_4x8__aarch32_neon);
}

TEST(Q8GEMM_4x8__AARCH32_NEON, k_eq_8_nozp) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .np(8)
      .kr(1)
      .m(4)
      .n(8)
      .k(8)
      .aZeroPoint(0)
      .bZeroPoint(0)
      .test(pytorch_q8gemm_ukernel_4x8__aarch32_neon);
}

TEST(Q8GEMM_4x8__AARCH32_NEON, k_gt_8) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester().mr(4).nr(8).np(8).kr(1).m(4).n(8).k(k).test(
        pytorch_q8gemm_ukernel_4x8__aarch32_neon);
  }
}

TEST(Q8GEMM_4x8__AARCH32_NEON, k_gt_8_strided_a) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .np(8)
        .kr(1)
        .m(4)
        .n(8)
        .k(k)
        .aStride(37)
        .test(pytorch_q8gemm_ukernel_4x8__aarch32_neon);
  }
}

TEST(Q8GEMM_4x8__AARCH32_NEON, k_gt_8_strided_c) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .np(8)
        .kr(1)
        .m(4)
        .n(8)
        .k(k)
        .cStride(17)
        .test(pytorch_q8gemm_ukernel_4x8__aarch32_neon);
  }
}

TEST(Q8GEMM_4x8__AARCH32_NEON, k_gt_8_azp0) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .np(8)
        .kr(1)
        .m(4)
        .n(8)
        .k(k)
        .aZeroPoint(0)
        .test(pytorch_q8gemm_ukernel_4x8__aarch32_neon);
  }
}

TEST(Q8GEMM_4x8__AARCH32_NEON, k_gt_8_bzp0) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .np(8)
        .kr(1)
        .m(4)
        .n(8)
        .k(k)
        .bZeroPoint(0)
        .test(pytorch_q8gemm_ukernel_4x8__aarch32_neon);
  }
}

TEST(Q8GEMM_4x8__AARCH32_NEON, k_gt_8_nozp) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .np(8)
        .kr(1)
        .m(4)
        .n(8)
        .k(k)
        .aZeroPoint(0)
        .bZeroPoint(0)
        .test(pytorch_q8gemm_ukernel_4x8__aarch32_neon);
  }
}

TEST(Q8GEMM_4x8__AARCH32_NEON, k_gt_8_subtile) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    for (uint32_t m = 1; m <= 4; m++) {
      for (uint32_t n = 1; n <= 8; n++) {
        GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .np(8)
            .kr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(3)
            .test(pytorch_q8gemm_ukernel_4x8__aarch32_neon);
      }
    }
  }
}

TEST(Q8GEMM_4x8__AARCH32_NEON, k_div_8) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 16; k < 128; k += 8) {
    GemmMicrokernelTester().mr(4).nr(8).np(8).kr(1).m(4).n(8).k(k).test(
        pytorch_q8gemm_ukernel_4x8__aarch32_neon);
  }
}

TEST(Q8GEMM_4x8__AARCH32_NEON, k_div_8_strided_a) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 16; k < 128; k += 8) {
    GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .np(8)
        .kr(1)
        .m(4)
        .n(8)
        .k(k)
        .aStride(171)
        .test(pytorch_q8gemm_ukernel_4x8__aarch32_neon);
  }
}

TEST(Q8GEMM_4x8__AARCH32_NEON, k_div_8_strided_c) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 16; k < 128; k += 8) {
    GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .np(8)
        .kr(1)
        .m(4)
        .n(8)
        .k(k)
        .cStride(17)
        .test(pytorch_q8gemm_ukernel_4x8__aarch32_neon);
  }
}

TEST(Q8GEMM_4x8__AARCH32_NEON, k_div_8_subtile) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 16; k < 128; k += 24) {
    for (uint32_t m = 1; m <= 4; m++) {
      for (uint32_t n = 1; n <= 8; n++) {
        GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .np(8)
            .kr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(3)
            .test(pytorch_q8gemm_ukernel_4x8__aarch32_neon);
      }
    }
  }
}

TEST(Q8GEMM_4x8c2_XZP__AARCH32_NEON, k_eq_8) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester().mr(4).nr(8).np(8).kr(2).m(4).n(8).k(8).test(
      pytorch_q8gemm_xzp_ukernel_4x8c2__aarch32_neon);
}

TEST(Q8GEMM_4x8c2_XZP__AARCH32_NEON, k_eq_8_strided_a) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .np(8)
      .kr(2)
      .m(4)
      .n(8)
      .k(8)
      .aStride(37)
      .test(pytorch_q8gemm_xzp_ukernel_4x8c2__aarch32_neon);
}

TEST(Q8GEMM_4x8c2_XZP__AARCH32_NEON, k_eq_8_strided_c) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .np(8)
      .kr(2)
      .m(4)
      .n(8)
      .k(8)
      .cStride(17)
      .test(pytorch_q8gemm_xzp_ukernel_4x8c2__aarch32_neon);
}

TEST(Q8GEMM_4x8c2_XZP__AARCH32_NEON, k_eq_8_qmin128) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester().mr(4).nr(8).np(8).kr(2).m(4).n(8).k(8).qmin(128).test(
      pytorch_q8gemm_xzp_ukernel_4x8c2__aarch32_neon);
}

TEST(Q8GEMM_4x8c2_XZP__AARCH32_NEON, k_eq_8_qmax128) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester().mr(4).nr(8).np(8).kr(2).m(4).n(8).k(8).qmax(128).test(
      pytorch_q8gemm_xzp_ukernel_4x8c2__aarch32_neon);
}

TEST(Q8GEMM_4x8c2_XZP__AARCH32_NEON, k_eq_8_azp0) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .np(8)
      .kr(2)
      .m(4)
      .n(8)
      .k(8)
      .aZeroPoint(0)
      .test(pytorch_q8gemm_xzp_ukernel_4x8c2__aarch32_neon);
}

TEST(Q8GEMM_4x8c2_XZP__AARCH32_NEON, k_eq_8_bzp0) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .np(8)
      .kr(2)
      .m(4)
      .n(8)
      .k(8)
      .bZeroPoint(0)
      .test(pytorch_q8gemm_xzp_ukernel_4x8c2__aarch32_neon);
}

TEST(Q8GEMM_4x8c2_XZP__AARCH32_NEON, k_eq_8_nozp) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .np(8)
      .kr(2)
      .m(4)
      .n(8)
      .k(8)
      .aZeroPoint(0)
      .bZeroPoint(0)
      .test(pytorch_q8gemm_xzp_ukernel_4x8c2__aarch32_neon);
}

TEST(Q8GEMM_4x8c2_XZP__AARCH32_NEON, k_gt_8) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester().mr(4).nr(8).np(8).kr(2).m(4).n(8).k(k).test(
        pytorch_q8gemm_xzp_ukernel_4x8c2__aarch32_neon);
  }
}

TEST(Q8GEMM_4x8c2_XZP__AARCH32_NEON, k_gt_8_strided_a) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .np(8)
        .kr(2)
        .m(4)
        .n(8)
        .k(k)
        .aStride(37)
        .test(pytorch_q8gemm_xzp_ukernel_4x8c2__aarch32_neon);
  }
}

TEST(Q8GEMM_4x8c2_XZP__AARCH32_NEON, k_gt_8_strided_c) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .np(8)
        .kr(2)
        .m(4)
        .n(8)
        .k(k)
        .cStride(17)
        .test(pytorch_q8gemm_xzp_ukernel_4x8c2__aarch32_neon);
  }
}

TEST(Q8GEMM_4x8c2_XZP__AARCH32_NEON, k_gt_8_azp0) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .np(8)
        .kr(2)
        .m(4)
        .n(8)
        .k(k)
        .aZeroPoint(0)
        .test(pytorch_q8gemm_xzp_ukernel_4x8c2__aarch32_neon);
  }
}

TEST(Q8GEMM_4x8c2_XZP__AARCH32_NEON, k_gt_8_bzp0) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .np(8)
        .kr(2)
        .m(4)
        .n(8)
        .k(k)
        .bZeroPoint(0)
        .test(pytorch_q8gemm_xzp_ukernel_4x8c2__aarch32_neon);
  }
}

TEST(Q8GEMM_4x8c2_XZP__AARCH32_NEON, k_gt_8_nozp) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .np(8)
        .kr(2)
        .m(4)
        .n(8)
        .k(k)
        .aZeroPoint(0)
        .bZeroPoint(0)
        .test(pytorch_q8gemm_xzp_ukernel_4x8c2__aarch32_neon);
  }
}

TEST(Q8GEMM_4x8c2_XZP__AARCH32_NEON, k_gt_8_subtile) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    for (uint32_t m = 1; m <= 4; m++) {
      for (uint32_t n = 1; n <= 8; n++) {
        GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .np(8)
            .kr(2)
            .m(m)
            .n(n)
            .k(k)
            .iterations(3)
            .test(pytorch_q8gemm_xzp_ukernel_4x8c2__aarch32_neon);
      }
    }
  }
}

TEST(Q8GEMM_4x8c2_XZP__AARCH32_NEON, k_div_8) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 16; k < 128; k += 8) {
    GemmMicrokernelTester().mr(4).nr(8).np(8).kr(2).m(4).n(8).k(k).test(
        pytorch_q8gemm_xzp_ukernel_4x8c2__aarch32_neon);
  }
}

TEST(Q8GEMM_4x8c2_XZP__AARCH32_NEON, k_div_8_strided_a) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 16; k < 128; k += 8) {
    GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .np(8)
        .kr(2)
        .m(4)
        .n(8)
        .k(k)
        .aStride(171)
        .test(pytorch_q8gemm_xzp_ukernel_4x8c2__aarch32_neon);
  }
}

TEST(Q8GEMM_4x8c2_XZP__AARCH32_NEON, k_div_8_strided_c) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 16; k < 128; k += 8) {
    GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .np(8)
        .kr(2)
        .m(4)
        .n(8)
        .k(k)
        .cStride(17)
        .test(pytorch_q8gemm_xzp_ukernel_4x8c2__aarch32_neon);
  }
}

TEST(Q8GEMM_4x8c2_XZP__AARCH32_NEON, k_div_8_subtile) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 16; k < 128; k += 24) {
    for (uint32_t m = 1; m <= 4; m++) {
      for (uint32_t n = 1; n <= 8; n++) {
        GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .np(8)
            .kr(2)
            .m(m)
            .n(n)
            .k(k)
            .iterations(3)
            .test(pytorch_q8gemm_xzp_ukernel_4x8c2__aarch32_neon);
      }
    }
  }
}
#endif

#if CPUINFO_ARCH_ARM64
TEST(Q8GEMM_8x8__AARCH64_NEON, k_eq_8) {
  GemmMicrokernelTester().mr(8).nr(8).np(8).kr(1).m(8).n(8).k(8).test(
      pytorch_q8gemm_ukernel_8x8__aarch64_neon);
}

TEST(Q8GEMM_8x8__AARCH64_NEON, k_eq_8_strided_a) {
  GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .np(8)
      .kr(1)
      .m(8)
      .n(8)
      .k(8)
      .aStride(37)
      .test(pytorch_q8gemm_ukernel_8x8__aarch64_neon);
}

TEST(Q8GEMM_8x8__AARCH64_NEON, k_eq_8_strided_c) {
  GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .np(8)
      .kr(1)
      .m(8)
      .n(8)
      .k(8)
      .cStride(17)
      .test(pytorch_q8gemm_ukernel_8x8__aarch64_neon);
}

TEST(Q8GEMM_8x8__AARCH64_NEON, k_eq_8_qmin128) {
  GemmMicrokernelTester().mr(8).nr(8).np(8).kr(1).m(8).n(8).k(8).qmin(128).test(
      pytorch_q8gemm_ukernel_8x8__aarch64_neon);
}

TEST(Q8GEMM_8x8__AARCH64_NEON, k_eq_8_qmax128) {
  GemmMicrokernelTester().mr(8).nr(8).np(8).kr(1).m(8).n(8).k(8).qmax(128).test(
      pytorch_q8gemm_ukernel_8x8__aarch64_neon);
}

TEST(Q8GEMM_8x8__AARCH64_NEON, k_eq_8_azp0) {
  GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .np(8)
      .kr(1)
      .m(8)
      .n(8)
      .k(8)
      .aZeroPoint(0)
      .test(pytorch_q8gemm_ukernel_8x8__aarch64_neon);
}

TEST(Q8GEMM_8x8__AARCH64_NEON, k_eq_8_bzp0) {
  GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .np(8)
      .kr(1)
      .m(8)
      .n(8)
      .k(8)
      .bZeroPoint(0)
      .test(pytorch_q8gemm_ukernel_8x8__aarch64_neon);
}

TEST(Q8GEMM_8x8__AARCH64_NEON, k_eq_8_nozp) {
  GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .np(8)
      .kr(1)
      .m(8)
      .n(8)
      .k(8)
      .aZeroPoint(0)
      .bZeroPoint(0)
      .test(pytorch_q8gemm_ukernel_8x8__aarch64_neon);
}

TEST(Q8GEMM_8x8__AARCH64_NEON, k_gt_8) {
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester().mr(8).nr(8).np(8).kr(1).m(8).n(8).k(k).test(
        pytorch_q8gemm_ukernel_8x8__aarch64_neon);
  }
}

TEST(Q8GEMM_8x8__AARCH64_NEON, k_gt_8_strided_a) {
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .np(8)
        .kr(1)
        .m(8)
        .n(8)
        .k(k)
        .aStride(37)
        .test(pytorch_q8gemm_ukernel_8x8__aarch64_neon);
  }
}

TEST(Q8GEMM_8x8__AARCH64_NEON, k_gt_8_strided_c) {
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .np(8)
        .kr(1)
        .m(8)
        .n(8)
        .k(k)
        .cStride(17)
        .test(pytorch_q8gemm_ukernel_8x8__aarch64_neon);
  }
}

TEST(Q8GEMM_8x8__AARCH64_NEON, k_gt_8_azp0) {
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .np(8)
        .kr(1)
        .m(8)
        .n(8)
        .k(k)
        .aZeroPoint(0)
        .test(pytorch_q8gemm_ukernel_8x8__aarch64_neon);
  }
}

TEST(Q8GEMM_8x8__AARCH64_NEON, k_gt_8_bzp0) {
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .np(8)
        .kr(1)
        .m(8)
        .n(8)
        .k(k)
        .bZeroPoint(0)
        .test(pytorch_q8gemm_ukernel_8x8__aarch64_neon);
  }
}

TEST(Q8GEMM_8x8__AARCH64_NEON, k_gt_8_nozp) {
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .np(8)
        .kr(1)
        .m(8)
        .n(8)
        .k(k)
        .aZeroPoint(0)
        .bZeroPoint(0)
        .test(pytorch_q8gemm_ukernel_8x8__aarch64_neon);
  }
}

TEST(Q8GEMM_8x8__AARCH64_NEON, k_gt_8_subtile) {
  for (size_t k = 9; k < 16; k++) {
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
            .test(pytorch_q8gemm_ukernel_8x8__aarch64_neon);
      }
    }
  }
}

TEST(Q8GEMM_8x8__AARCH64_NEON, k_div_8) {
  for (size_t k = 16; k < 128; k += 8) {
    GemmMicrokernelTester().mr(8).nr(8).np(8).kr(1).m(8).n(8).k(k).test(
        pytorch_q8gemm_ukernel_8x8__aarch64_neon);
  }
}

TEST(Q8GEMM_8x8__AARCH64_NEON, k_div_8_strided_a) {
  for (size_t k = 16; k < 128; k += 8) {
    GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .np(8)
        .kr(1)
        .m(8)
        .n(8)
        .k(k)
        .aStride(171)
        .test(pytorch_q8gemm_ukernel_8x8__aarch64_neon);
  }
}

TEST(Q8GEMM_8x8__AARCH64_NEON, k_div_8_strided_c) {
  for (size_t k = 16; k < 128; k += 8) {
    GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .np(8)
        .kr(1)
        .m(8)
        .n(8)
        .k(k)
        .cStride(17)
        .test(pytorch_q8gemm_ukernel_8x8__aarch64_neon);
  }
}

TEST(Q8GEMM_8x8__AARCH64_NEON, k_div_8_subtile) {
  for (size_t k = 16; k < 128; k += 24) {
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
            .test(pytorch_q8gemm_ukernel_8x8__aarch64_neon);
      }
    }
  }
}
#endif

#if CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64
TEST(Q8GEMM_4x8__NEON, k_eq_8) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester().mr(4).nr(8).np(8).kr(1).m(4).n(8).k(8).test(
      pytorch_q8gemm_ukernel_4x8__neon);
}

TEST(Q8GEMM_4x8__NEON, k_eq_8_strided_a) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .np(8)
      .kr(1)
      .m(4)
      .n(8)
      .k(8)
      .aStride(37)
      .test(pytorch_q8gemm_ukernel_4x8__neon);
}

TEST(Q8GEMM_4x8__NEON, k_eq_8_strided_c) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .np(8)
      .kr(1)
      .m(4)
      .n(8)
      .k(8)
      .cStride(17)
      .test(pytorch_q8gemm_ukernel_4x8__neon);
}

TEST(Q8GEMM_4x8__NEON, k_eq_8_qmin128) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester().mr(4).nr(8).np(8).kr(1).m(4).n(8).k(8).qmin(128).test(
      pytorch_q8gemm_ukernel_4x8__neon);
}

TEST(Q8GEMM_4x8__NEON, k_eq_8_qmax128) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester().mr(4).nr(8).np(8).kr(1).m(4).n(8).k(8).qmax(128).test(
      pytorch_q8gemm_ukernel_4x8__neon);
}

TEST(Q8GEMM_4x8__NEON, k_eq_8_azp0) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .np(8)
      .kr(1)
      .m(4)
      .n(8)
      .k(8)
      .aZeroPoint(0)
      .test(pytorch_q8gemm_ukernel_4x8__neon);
}

TEST(Q8GEMM_4x8__NEON, k_eq_8_bzp0) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .np(8)
      .kr(1)
      .m(4)
      .n(8)
      .k(8)
      .bZeroPoint(0)
      .test(pytorch_q8gemm_ukernel_4x8__neon);
}

TEST(Q8GEMM_4x8__NEON, k_eq_8_nozp) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .np(8)
      .kr(1)
      .m(4)
      .n(8)
      .k(8)
      .aZeroPoint(0)
      .bZeroPoint(0)
      .test(pytorch_q8gemm_ukernel_4x8__neon);
}

TEST(Q8GEMM_4x8__NEON, k_gt_8) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester().mr(4).nr(8).np(8).kr(1).m(4).n(8).k(k).test(
        pytorch_q8gemm_ukernel_4x8__neon);
  }
}

TEST(Q8GEMM_4x8__NEON, k_gt_8_strided_a) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .np(8)
        .kr(1)
        .m(4)
        .n(8)
        .k(k)
        .aStride(37)
        .test(pytorch_q8gemm_ukernel_4x8__neon);
  }
}

TEST(Q8GEMM_4x8__NEON, k_gt_8_strided_c) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .np(8)
        .kr(1)
        .m(4)
        .n(8)
        .k(k)
        .cStride(17)
        .test(pytorch_q8gemm_ukernel_4x8__neon);
  }
}

TEST(Q8GEMM_4x8__NEON, k_gt_8_azp0) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .np(8)
        .kr(1)
        .m(4)
        .n(8)
        .k(k)
        .aZeroPoint(0)
        .test(pytorch_q8gemm_ukernel_4x8__neon);
  }
}

TEST(Q8GEMM_4x8__NEON, k_gt_8_bzp0) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .np(8)
        .kr(1)
        .m(4)
        .n(8)
        .k(k)
        .bZeroPoint(0)
        .test(pytorch_q8gemm_ukernel_4x8__neon);
  }
}

TEST(Q8GEMM_4x8__NEON, k_gt_8_nozp) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .np(8)
        .kr(1)
        .m(4)
        .n(8)
        .k(k)
        .aZeroPoint(0)
        .bZeroPoint(0)
        .test(pytorch_q8gemm_ukernel_4x8__neon);
  }
}

TEST(Q8GEMM_4x8__NEON, k_gt_8_subtile) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    for (uint32_t m = 1; m <= 4; m++) {
      for (uint32_t n = 1; n <= 8; n++) {
        GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .np(8)
            .kr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(3)
            .test(pytorch_q8gemm_ukernel_4x8__neon);
      }
    }
  }
}

TEST(Q8GEMM_4x8__NEON, k_div_8) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 16; k < 128; k += 8) {
    GemmMicrokernelTester().mr(4).nr(8).np(8).kr(1).m(4).n(8).k(k).test(
        pytorch_q8gemm_ukernel_4x8__neon);
  }
}

TEST(Q8GEMM_4x8__NEON, k_div_8_strided_a) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 16; k < 128; k += 8) {
    GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .np(8)
        .kr(1)
        .m(4)
        .n(8)
        .k(k)
        .aStride(171)
        .test(pytorch_q8gemm_ukernel_4x8__neon);
  }
}

TEST(Q8GEMM_4x8__NEON, k_div_8_strided_c) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 16; k < 128; k += 8) {
    GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .np(8)
        .kr(1)
        .m(4)
        .n(8)
        .k(k)
        .cStride(17)
        .test(pytorch_q8gemm_ukernel_4x8__neon);
  }
}

TEST(Q8GEMM_4x8__NEON, k_div_8_subtile) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 16; k < 128; k += 24) {
    for (uint32_t m = 1; m <= 4; m++) {
      for (uint32_t n = 1; n <= 8; n++) {
        GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .np(8)
            .kr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(3)
            .test(pytorch_q8gemm_ukernel_4x8__neon);
      }
    }
  }
}

TEST(Q8GEMM_8x8__NEON, k_eq_8) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester().mr(8).nr(8).np(8).kr(1).m(8).n(8).k(8).test(
      pytorch_q8gemm_ukernel_8x8__neon);
}

TEST(Q8GEMM_8x8__NEON, k_eq_8_strided_a) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .np(8)
      .kr(1)
      .m(8)
      .n(8)
      .k(8)
      .aStride(37)
      .test(pytorch_q8gemm_ukernel_8x8__neon);
}

TEST(Q8GEMM_8x8__NEON, k_eq_8_strided_c) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .np(8)
      .kr(1)
      .m(8)
      .n(8)
      .k(8)
      .cStride(17)
      .test(pytorch_q8gemm_ukernel_8x8__neon);
}

TEST(Q8GEMM_8x8__NEON, k_eq_8_qmin128) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester().mr(8).nr(8).np(8).kr(1).m(8).n(8).k(8).qmin(128).test(
      pytorch_q8gemm_ukernel_8x8__neon);
}

TEST(Q8GEMM_8x8__NEON, k_eq_8_qmax128) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester().mr(8).nr(8).np(8).kr(1).m(8).n(8).k(8).qmax(128).test(
      pytorch_q8gemm_ukernel_8x8__neon);
}

TEST(Q8GEMM_8x8__NEON, k_eq_8_azp0) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .np(8)
      .kr(1)
      .m(8)
      .n(8)
      .k(8)
      .aZeroPoint(0)
      .test(pytorch_q8gemm_ukernel_8x8__neon);
}

TEST(Q8GEMM_8x8__NEON, k_eq_8_bzp0) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .np(8)
      .kr(1)
      .m(8)
      .n(8)
      .k(8)
      .bZeroPoint(0)
      .test(pytorch_q8gemm_ukernel_8x8__neon);
}

TEST(Q8GEMM_8x8__NEON, k_eq_8_nozp) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .np(8)
      .kr(1)
      .m(8)
      .n(8)
      .k(8)
      .aZeroPoint(0)
      .bZeroPoint(0)
      .test(pytorch_q8gemm_ukernel_8x8__neon);
}

TEST(Q8GEMM_8x8__NEON, k_gt_8) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester().mr(8).nr(8).np(8).kr(1).m(8).n(8).k(k).test(
        pytorch_q8gemm_ukernel_8x8__neon);
  }
}

TEST(Q8GEMM_8x8__NEON, k_gt_8_strided_a) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .np(8)
        .kr(1)
        .m(8)
        .n(8)
        .k(k)
        .aStride(37)
        .test(pytorch_q8gemm_ukernel_8x8__neon);
  }
}

TEST(Q8GEMM_8x8__NEON, k_gt_8_strided_c) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .np(8)
        .kr(1)
        .m(8)
        .n(8)
        .k(k)
        .cStride(17)
        .test(pytorch_q8gemm_ukernel_8x8__neon);
  }
}

TEST(Q8GEMM_8x8__NEON, k_gt_8_azp0) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .np(8)
        .kr(1)
        .m(8)
        .n(8)
        .k(k)
        .aZeroPoint(0)
        .test(pytorch_q8gemm_ukernel_8x8__neon);
  }
}

TEST(Q8GEMM_8x8__NEON, k_gt_8_bzp0) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .np(8)
        .kr(1)
        .m(8)
        .n(8)
        .k(k)
        .bZeroPoint(0)
        .test(pytorch_q8gemm_ukernel_8x8__neon);
  }
}

TEST(Q8GEMM_8x8__NEON, k_gt_8_nozp) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .np(8)
        .kr(1)
        .m(8)
        .n(8)
        .k(k)
        .aZeroPoint(0)
        .bZeroPoint(0)
        .test(pytorch_q8gemm_ukernel_8x8__neon);
  }
}

TEST(Q8GEMM_8x8__NEON, k_gt_8_subtile) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
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
            .test(pytorch_q8gemm_ukernel_8x8__neon);
      }
    }
  }
}

TEST(Q8GEMM_8x8__NEON, k_div_8) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 16; k < 128; k += 8) {
    GemmMicrokernelTester().mr(8).nr(8).np(8).kr(1).m(8).n(8).k(k).test(
        pytorch_q8gemm_ukernel_8x8__neon);
  }
}

TEST(Q8GEMM_8x8__NEON, k_div_8_strided_a) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 16; k < 128; k += 8) {
    GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .np(8)
        .kr(1)
        .m(8)
        .n(8)
        .k(k)
        .aStride(171)
        .test(pytorch_q8gemm_ukernel_8x8__neon);
  }
}

TEST(Q8GEMM_8x8__NEON, k_div_8_strided_c) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 16; k < 128; k += 8) {
    GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .np(8)
        .kr(1)
        .m(8)
        .n(8)
        .k(k)
        .cStride(17)
        .test(pytorch_q8gemm_ukernel_8x8__neon);
  }
}

TEST(Q8GEMM_8x8__NEON, k_div_8_subtile) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 16; k < 128; k += 24) {
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
            .test(pytorch_q8gemm_ukernel_8x8__neon);
      }
    }
  }
}

TEST(Q8GEMM_6x4__NEON, k_eq_8) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester().mr(6).nr(4).np(4).kr(1).m(6).n(4).k(8).test(
      pytorch_q8gemm_ukernel_6x4__neon);
}

TEST(Q8GEMM_6x4__NEON, k_eq_8_strided_a) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester()
      .mr(6)
      .nr(4)
      .np(4)
      .kr(1)
      .m(6)
      .n(4)
      .k(8)
      .aStride(37)
      .test(pytorch_q8gemm_ukernel_6x4__neon);
}

TEST(Q8GEMM_6x4__NEON, k_eq_8_strided_c) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester()
      .mr(6)
      .nr(4)
      .np(4)
      .kr(1)
      .m(6)
      .n(4)
      .k(8)
      .cStride(17)
      .test(pytorch_q8gemm_ukernel_6x4__neon);
}

TEST(Q8GEMM_6x4__NEON, k_eq_8_qmin128) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester().mr(6).nr(4).np(4).kr(1).m(6).n(4).k(8).qmin(128).test(
      pytorch_q8gemm_ukernel_6x4__neon);
}

TEST(Q8GEMM_6x4__NEON, k_eq_8_qmax128) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester().mr(6).nr(4).np(4).kr(1).m(6).n(4).k(8).qmax(128).test(
      pytorch_q8gemm_ukernel_6x4__neon);
}

TEST(Q8GEMM_6x4__NEON, k_eq_8_azp0) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester()
      .mr(6)
      .nr(4)
      .np(4)
      .kr(1)
      .m(6)
      .n(4)
      .k(8)
      .aZeroPoint(0)
      .test(pytorch_q8gemm_ukernel_6x4__neon);
}

TEST(Q8GEMM_6x4__NEON, k_eq_8_bzp0) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester()
      .mr(6)
      .nr(4)
      .np(4)
      .kr(1)
      .m(6)
      .n(4)
      .k(8)
      .bZeroPoint(0)
      .test(pytorch_q8gemm_ukernel_6x4__neon);
}

TEST(Q8GEMM_6x4__NEON, k_eq_8_nozp) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester()
      .mr(6)
      .nr(4)
      .np(4)
      .kr(1)
      .m(6)
      .n(4)
      .k(8)
      .aZeroPoint(0)
      .bZeroPoint(0)
      .test(pytorch_q8gemm_ukernel_6x4__neon);
}

TEST(Q8GEMM_6x4__NEON, k_gt_8) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester().mr(6).nr(4).np(4).kr(1).m(6).n(4).k(k).test(
        pytorch_q8gemm_ukernel_6x4__neon);
  }
}

TEST(Q8GEMM_6x4__NEON, k_gt_8_strided_a) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(6)
        .nr(4)
        .np(4)
        .kr(1)
        .m(6)
        .n(4)
        .k(k)
        .aStride(37)
        .test(pytorch_q8gemm_ukernel_6x4__neon);
  }
}

TEST(Q8GEMM_6x4__NEON, k_gt_8_strided_c) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(6)
        .nr(4)
        .np(4)
        .kr(1)
        .m(6)
        .n(4)
        .k(k)
        .cStride(17)
        .test(pytorch_q8gemm_ukernel_6x4__neon);
  }
}

TEST(Q8GEMM_6x4__NEON, k_gt_8_azp0) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(6)
        .nr(4)
        .np(4)
        .kr(1)
        .m(6)
        .n(4)
        .k(k)
        .aZeroPoint(0)
        .test(pytorch_q8gemm_ukernel_6x4__neon);
  }
}

TEST(Q8GEMM_6x4__NEON, k_gt_8_bzp0) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(6)
        .nr(4)
        .np(4)
        .kr(1)
        .m(6)
        .n(4)
        .k(k)
        .bZeroPoint(0)
        .test(pytorch_q8gemm_ukernel_6x4__neon);
  }
}

TEST(Q8GEMM_6x4__NEON, k_gt_8_nozp) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(6)
        .nr(4)
        .np(4)
        .kr(1)
        .m(6)
        .n(4)
        .k(k)
        .aZeroPoint(0)
        .bZeroPoint(0)
        .test(pytorch_q8gemm_ukernel_6x4__neon);
  }
}

TEST(Q8GEMM_6x4__NEON, k_gt_8_subtile) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    for (uint32_t m = 1; m <= 6; m++) {
      for (uint32_t n = 1; n <= 4; n++) {
        GemmMicrokernelTester()
            .mr(6)
            .nr(4)
            .np(4)
            .kr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(3)
            .test(pytorch_q8gemm_ukernel_6x4__neon);
      }
    }
  }
}

TEST(Q8GEMM_6x4__NEON, k_div_8) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 16; k < 128; k += 8) {
    GemmMicrokernelTester().mr(6).nr(4).np(4).kr(1).m(6).n(4).k(k).test(
        pytorch_q8gemm_ukernel_6x4__neon);
  }
}

TEST(Q8GEMM_6x4__NEON, k_div_8_strided_a) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 16; k < 128; k += 8) {
    GemmMicrokernelTester()
        .mr(6)
        .nr(4)
        .np(4)
        .kr(1)
        .m(6)
        .n(4)
        .k(k)
        .aStride(171)
        .test(pytorch_q8gemm_ukernel_6x4__neon);
  }
}

TEST(Q8GEMM_6x4__NEON, k_div_8_strided_c) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 16; k < 128; k += 8) {
    GemmMicrokernelTester()
        .mr(6)
        .nr(4)
        .np(4)
        .kr(1)
        .m(6)
        .n(4)
        .k(k)
        .cStride(17)
        .test(pytorch_q8gemm_ukernel_6x4__neon);
  }
}

TEST(Q8GEMM_6x4__NEON, k_div_8_subtile) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 16; k < 128; k += 24) {
    for (uint32_t m = 1; m <= 6; m++) {
      for (uint32_t n = 1; n <= 4; n++) {
        GemmMicrokernelTester().mr(6).nr(4).np(4).kr(1).m(m).n(n).k(k).test(
            pytorch_q8gemm_ukernel_6x4__neon);
      }
    }
  }
}

TEST(Q8GEMM_4x8c2_XZP__NEON, k_eq_8) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester().mr(4).nr(8).np(8).kr(2).m(4).n(8).k(8).test(
      pytorch_q8gemm_xzp_ukernel_4x8c2__neon);
}

TEST(Q8GEMM_4x8c2_XZP__NEON, k_eq_8_strided_a) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .np(8)
      .kr(2)
      .m(4)
      .n(8)
      .k(8)
      .aStride(37)
      .test(pytorch_q8gemm_xzp_ukernel_4x8c2__neon);
}

TEST(Q8GEMM_4x8c2_XZP__NEON, k_eq_8_strided_c) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .np(8)
      .kr(2)
      .m(4)
      .n(8)
      .k(8)
      .cStride(17)
      .test(pytorch_q8gemm_xzp_ukernel_4x8c2__neon);
}

TEST(Q8GEMM_4x8c2_XZP__NEON, k_eq_8_qmin128) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester().mr(4).nr(8).np(8).kr(2).m(4).n(8).k(8).qmin(128).test(
      pytorch_q8gemm_xzp_ukernel_4x8c2__neon);
}

TEST(Q8GEMM_4x8c2_XZP__NEON, k_eq_8_qmax128) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester().mr(4).nr(8).np(8).kr(2).m(4).n(8).k(8).qmax(128).test(
      pytorch_q8gemm_xzp_ukernel_4x8c2__neon);
}

TEST(Q8GEMM_4x8c2_XZP__NEON, k_eq_8_azp0) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .np(8)
      .kr(2)
      .m(4)
      .n(8)
      .k(8)
      .aZeroPoint(0)
      .test(pytorch_q8gemm_xzp_ukernel_4x8c2__neon);
}

TEST(Q8GEMM_4x8c2_XZP__NEON, k_eq_8_bzp0) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .np(8)
      .kr(2)
      .m(4)
      .n(8)
      .k(8)
      .bZeroPoint(0)
      .test(pytorch_q8gemm_xzp_ukernel_4x8c2__neon);
}

TEST(Q8GEMM_4x8c2_XZP__NEON, k_eq_8_nozp) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .np(8)
      .kr(2)
      .m(4)
      .n(8)
      .k(8)
      .aZeroPoint(0)
      .bZeroPoint(0)
      .test(pytorch_q8gemm_xzp_ukernel_4x8c2__neon);
}

TEST(Q8GEMM_4x8c2_XZP__NEON, k_gt_8) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester().mr(4).nr(8).np(8).kr(2).m(4).n(8).k(k).test(
        pytorch_q8gemm_xzp_ukernel_4x8c2__neon);
  }
}

TEST(Q8GEMM_4x8c2_XZP__NEON, k_gt_8_strided_a) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .np(8)
        .kr(2)
        .m(4)
        .n(8)
        .k(k)
        .aStride(37)
        .test(pytorch_q8gemm_xzp_ukernel_4x8c2__neon);
  }
}

TEST(Q8GEMM_4x8c2_XZP__NEON, k_gt_8_strided_c) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .np(8)
        .kr(2)
        .m(4)
        .n(8)
        .k(k)
        .cStride(17)
        .test(pytorch_q8gemm_xzp_ukernel_4x8c2__neon);
  }
}

TEST(Q8GEMM_4x8c2_XZP__NEON, k_gt_8_azp0) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .np(8)
        .kr(2)
        .m(4)
        .n(8)
        .k(k)
        .aZeroPoint(0)
        .test(pytorch_q8gemm_xzp_ukernel_4x8c2__neon);
  }
}

TEST(Q8GEMM_4x8c2_XZP__NEON, k_gt_8_bzp0) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .np(8)
        .kr(2)
        .m(4)
        .n(8)
        .k(k)
        .bZeroPoint(0)
        .test(pytorch_q8gemm_xzp_ukernel_4x8c2__neon);
  }
}

TEST(Q8GEMM_4x8c2_XZP__NEON, k_gt_8_nozp) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .np(8)
        .kr(2)
        .m(4)
        .n(8)
        .k(k)
        .aZeroPoint(0)
        .bZeroPoint(0)
        .test(pytorch_q8gemm_xzp_ukernel_4x8c2__neon);
  }
}

TEST(Q8GEMM_4x8c2_XZP__NEON, k_gt_8_subtile) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 9; k < 16; k++) {
    for (uint32_t m = 1; m <= 4; m++) {
      for (uint32_t n = 1; n <= 8; n++) {
        GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .np(8)
            .kr(2)
            .m(m)
            .n(n)
            .k(k)
            .iterations(3)
            .test(pytorch_q8gemm_xzp_ukernel_4x8c2__neon);
      }
    }
  }
}

TEST(Q8GEMM_4x8c2_XZP__NEON, k_div_8) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 16; k < 128; k += 8) {
    GemmMicrokernelTester().mr(4).nr(8).np(8).kr(2).m(4).n(8).k(k).test(
        pytorch_q8gemm_xzp_ukernel_4x8c2__neon);
  }
}

TEST(Q8GEMM_4x8c2_XZP__NEON, k_div_8_strided_a) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 16; k < 128; k += 8) {
    GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .np(8)
        .kr(2)
        .m(4)
        .n(8)
        .k(k)
        .aStride(171)
        .test(pytorch_q8gemm_xzp_ukernel_4x8c2__neon);
  }
}

TEST(Q8GEMM_4x8c2_XZP__NEON, k_div_8_strided_c) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 16; k < 128; k += 8) {
    GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .np(8)
        .kr(2)
        .m(4)
        .n(8)
        .k(k)
        .cStride(17)
        .test(pytorch_q8gemm_xzp_ukernel_4x8c2__neon);
  }
}

TEST(Q8GEMM_4x8c2_XZP__NEON, k_div_8_subtile) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t k = 16; k < 128; k += 24) {
    for (uint32_t m = 1; m <= 4; m++) {
      for (uint32_t n = 1; n <= 8; n++) {
        GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .np(8)
            .kr(2)
            .m(m)
            .n(n)
            .k(k)
            .iterations(3)
            .test(pytorch_q8gemm_xzp_ukernel_4x8c2__neon);
      }
    }
  }
}
#endif

#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
TEST(Q8GEMM_2x4c8__SSE2, k_eq_8) {
  TEST_REQUIRES_X86_SSE2;
  GemmMicrokernelTester().mr(2).nr(4).np(1).kr(8).m(2).n(4).k(8).test(
      pytorch_q8gemm_ukernel_2x4c8__sse2);
}

TEST(Q8GEMM_2x4c8__SSE2, k_eq_8_strided_a) {
  TEST_REQUIRES_X86_SSE2;
  GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .np(1)
      .kr(8)
      .m(2)
      .n(4)
      .k(8)
      .aStride(37)
      .test(pytorch_q8gemm_ukernel_2x4c8__sse2);
}

TEST(Q8GEMM_2x4c8__SSE2, k_eq_8_strided_c) {
  TEST_REQUIRES_X86_SSE2;
  GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .np(1)
      .kr(8)
      .m(2)
      .n(4)
      .k(8)
      .cStride(17)
      .test(pytorch_q8gemm_ukernel_2x4c8__sse2);
}

TEST(Q8GEMM_2x4c8__SSE2, k_eq_8_qmin128) {
  TEST_REQUIRES_X86_SSE2;
  GemmMicrokernelTester().mr(2).nr(4).np(1).kr(8).m(2).n(4).k(8).qmin(128).test(
      pytorch_q8gemm_ukernel_2x4c8__sse2);
}

TEST(Q8GEMM_2x4c8__SSE2, k_eq_8_qmax128) {
  TEST_REQUIRES_X86_SSE2;
  GemmMicrokernelTester().mr(2).nr(4).np(1).kr(8).m(2).n(4).k(8).qmax(128).test(
      pytorch_q8gemm_ukernel_2x4c8__sse2);
}

TEST(Q8GEMM_2x4c8__SSE2, k_eq_8_azp0) {
  TEST_REQUIRES_X86_SSE2;
  GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .np(1)
      .kr(8)
      .m(2)
      .n(4)
      .k(8)
      .aZeroPoint(0)
      .test(pytorch_q8gemm_ukernel_2x4c8__sse2);
}

TEST(Q8GEMM_2x4c8__SSE2, k_eq_8_bzp0) {
  TEST_REQUIRES_X86_SSE2;
  GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .np(1)
      .kr(8)
      .m(2)
      .n(4)
      .k(8)
      .bZeroPoint(0)
      .test(pytorch_q8gemm_ukernel_2x4c8__sse2);
}

TEST(Q8GEMM_2x4c8__SSE2, k_eq_8_nozp) {
  TEST_REQUIRES_X86_SSE2;
  GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .np(1)
      .kr(8)
      .m(2)
      .n(4)
      .k(8)
      .aZeroPoint(0)
      .bZeroPoint(0)
      .test(pytorch_q8gemm_ukernel_2x4c8__sse2);
}

TEST(Q8GEMM_2x4c8__SSE2, k_gt_8) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester().mr(2).nr(4).np(1).kr(8).m(2).n(4).k(k).test(
        pytorch_q8gemm_ukernel_2x4c8__sse2);
  }
}

TEST(Q8GEMM_2x4c8__SSE2, k_gt_8_strided_a) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .np(1)
        .kr(8)
        .m(2)
        .n(4)
        .k(k)
        .aStride(37)
        .test(pytorch_q8gemm_ukernel_2x4c8__sse2);
  }
}

TEST(Q8GEMM_2x4c8__SSE2, k_gt_8_strided_c) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .np(1)
        .kr(8)
        .m(2)
        .n(4)
        .k(k)
        .cStride(17)
        .test(pytorch_q8gemm_ukernel_2x4c8__sse2);
  }
}

TEST(Q8GEMM_2x4c8__SSE2, k_gt_8_azp0) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .np(1)
        .kr(8)
        .m(2)
        .n(4)
        .k(k)
        .aZeroPoint(0)
        .test(pytorch_q8gemm_ukernel_2x4c8__sse2);
  }
}

TEST(Q8GEMM_2x4c8__SSE2, k_gt_8_bzp0) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .np(1)
        .kr(8)
        .m(2)
        .n(4)
        .k(k)
        .bZeroPoint(0)
        .test(pytorch_q8gemm_ukernel_2x4c8__sse2);
  }
}

TEST(Q8GEMM_2x4c8__SSE2, k_gt_8_nozp) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .np(1)
        .kr(8)
        .m(2)
        .n(4)
        .k(k)
        .aZeroPoint(0)
        .bZeroPoint(0)
        .test(pytorch_q8gemm_ukernel_2x4c8__sse2);
  }
}

TEST(Q8GEMM_2x4c8__SSE2, k_gt_8_subtile) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t k = 9; k < 16; k++) {
    for (uint32_t m = 1; m <= 2; m++) {
      for (uint32_t n = 1; n <= 4; n++) {
        GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .np(1)
            .kr(8)
            .m(m)
            .n(n)
            .k(k)
            .iterations(3)
            .test(pytorch_q8gemm_ukernel_2x4c8__sse2);
      }
    }
  }
}

TEST(Q8GEMM_2x4c8__SSE2, k_div_8) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t k = 16; k < 128; k += 8) {
    GemmMicrokernelTester().mr(2).nr(4).np(1).kr(8).m(2).n(4).k(k).test(
        pytorch_q8gemm_ukernel_2x4c8__sse2);
  }
}

TEST(Q8GEMM_2x4c8__SSE2, k_div_8_strided_a) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t k = 16; k < 128; k += 8) {
    GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .np(1)
        .kr(8)
        .m(2)
        .n(4)
        .k(k)
        .aStride(171)
        .test(pytorch_q8gemm_ukernel_2x4c8__sse2);
  }
}

TEST(Q8GEMM_2x4c8__SSE2, k_div_8_strided_c) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t k = 16; k < 128; k += 8) {
    GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .np(1)
        .kr(8)
        .m(2)
        .n(4)
        .k(k)
        .cStride(17)
        .test(pytorch_q8gemm_ukernel_2x4c8__sse2);
  }
}

TEST(Q8GEMM_2x4c8__SSE2, k_div_8_subtile) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t k = 16; k < 128; k += 24) {
    for (uint32_t m = 1; m <= 2; m++) {
      for (uint32_t n = 1; n <= 4; n++) {
        GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .np(1)
            .kr(8)
            .m(m)
            .n(n)
            .k(k)
            .iterations(3)
            .test(pytorch_q8gemm_ukernel_2x4c8__sse2);
      }
    }
  }
}

// Following tests fail both on original QNNPack and the version
// with runtime requantization.

#if 0
  TEST(Q8GEMM_4x4c2__SSE2, k_eq_1) {
    TEST_REQUIRES_X86_SSE2;
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .np(4)
      .kr(2)
      .m(4)
      .n(4)
      .k(1)
      .test(pytorch_q8gemm_ukernel_4x4c2__sse2);
  }

  TEST(Q8GEMM_4x4c2__SSE2, k_eq_1_strided_a) {
    TEST_REQUIRES_X86_SSE2;
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .np(4)
      .kr(2)
      .m(4)
      .n(4)
      .k(1)
      .aStride(37)
      .test(pytorch_q8gemm_ukernel_4x4c2__sse2);
  }

  TEST(Q8GEMM_4x4c2__SSE2, k_eq_1_strided_c) {
    TEST_REQUIRES_X86_SSE2;
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .np(4)
      .kr(2)
      .m(4)
      .n(4)
      .k(1)
      .cStride(17)
      .test(pytorch_q8gemm_ukernel_4x4c2__sse2);
  }

  TEST(Q8GEMM_4x4c2__SSE2, k_eq_1_qmin128) {
    TEST_REQUIRES_X86_SSE2;
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .np(4)
      .kr(2)
      .m(4)
      .n(4)
      .k(1)
      .qmin(128)
      .test(pytorch_q8gemm_ukernel_4x4c2__sse2);
  }

  TEST(Q8GEMM_4x4c2__SSE2, k_eq_1_qmax128) {
    TEST_REQUIRES_X86_SSE2;
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .np(4)
      .kr(2)
      .m(4)
      .n(4)
      .k(1)
      .qmax(128)
      .test(pytorch_q8gemm_ukernel_4x4c2__sse2);
  }

  TEST(Q8GEMM_4x4c2__SSE2, k_eq_1_azp0) {
    TEST_REQUIRES_X86_SSE2;
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .np(4)
      .kr(2)
      .m(4)
      .n(4)
      .k(1)
      .aZeroPoint(0)
      .test(pytorch_q8gemm_ukernel_4x4c2__sse2);
  }

  TEST(Q8GEMM_4x4c2__SSE2, k_eq_1_bzp0) {
    TEST_REQUIRES_X86_SSE2;
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .np(4)
      .kr(2)
      .m(4)
      .n(4)
      .k(1)
      .bZeroPoint(0)
      .test(pytorch_q8gemm_ukernel_4x4c2__sse2);
  }

  TEST(Q8GEMM_4x4c2__SSE2, k_eq_1_nozp) {
    TEST_REQUIRES_X86_SSE2;
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .np(4)
      .kr(2)
      .m(4)
      .n(4)
      .k(1)
      .aZeroPoint(0)
      .bZeroPoint(0)
      .test(pytorch_q8gemm_ukernel_4x4c2__sse2);
  }
#endif

TEST(Q8GEMM_4x4c2__SSE2, k_lt_4) {
  TEST_REQUIRES_X86_SSE2;
  GemmMicrokernelTester().mr(4).nr(4).np(4).kr(2).m(4).n(4).k(3).test(
      pytorch_q8gemm_ukernel_4x4c2__sse2);
}

TEST(Q8GEMM_4x4c2__SSE2, k_lt_4_strided_a) {
  TEST_REQUIRES_X86_SSE2;
  GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .np(4)
      .kr(2)
      .m(4)
      .n(4)
      .k(3)
      .aStride(37)
      .test(pytorch_q8gemm_ukernel_4x4c2__sse2);
}

TEST(Q8GEMM_4x4c2__SSE2, k_lt_4_strided_c) {
  TEST_REQUIRES_X86_SSE2;
  GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .np(4)
      .kr(2)
      .m(4)
      .n(4)
      .k(3)
      .cStride(17)
      .test(pytorch_q8gemm_ukernel_4x4c2__sse2);
}

TEST(Q8GEMM_4x4c2__SSE2, k_lt_4_qmin128) {
  TEST_REQUIRES_X86_SSE2;
  GemmMicrokernelTester().mr(4).nr(4).np(4).kr(2).m(4).n(4).k(3).qmin(128).test(
      pytorch_q8gemm_ukernel_4x4c2__sse2);
}

TEST(Q8GEMM_4x4c2__SSE2, k_lt_4_qmax128) {
  TEST_REQUIRES_X86_SSE2;
  GemmMicrokernelTester().mr(4).nr(4).np(4).kr(2).m(4).n(4).k(3).qmax(128).test(
      pytorch_q8gemm_ukernel_4x4c2__sse2);
}

TEST(Q8GEMM_4x4c2__SSE2, k_lt_4_azp0) {
  TEST_REQUIRES_X86_SSE2;
  GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .np(4)
      .kr(2)
      .m(4)
      .n(4)
      .k(3)
      .aZeroPoint(0)
      .test(pytorch_q8gemm_ukernel_4x4c2__sse2);
}

TEST(Q8GEMM_4x4c2__SSE2, k_lt_4_bzp0) {
  TEST_REQUIRES_X86_SSE2;
  GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .np(4)
      .kr(2)
      .m(4)
      .n(4)
      .k(3)
      .bZeroPoint(0)
      .test(pytorch_q8gemm_ukernel_4x4c2__sse2);
}

TEST(Q8GEMM_4x4c2__SSE2, k_lt_4_nozp) {
  TEST_REQUIRES_X86_SSE2;
  GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .np(4)
      .kr(2)
      .m(4)
      .n(4)
      .k(3)
      .aZeroPoint(0)
      .bZeroPoint(0)
      .test(pytorch_q8gemm_ukernel_4x4c2__sse2);
}

TEST(Q8GEMM_4x4c2__SSE2, k_lt_8) {
  TEST_REQUIRES_X86_SSE2;
  GemmMicrokernelTester().mr(4).nr(4).np(4).kr(2).m(4).n(4).k(5).test(
      pytorch_q8gemm_ukernel_4x4c2__sse2);
}

TEST(Q8GEMM_4x4c2__SSE2, k_lt_8_strided_a) {
  TEST_REQUIRES_X86_SSE2;
  GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .np(4)
      .kr(2)
      .m(4)
      .n(4)
      .k(5)
      .aStride(37)
      .test(pytorch_q8gemm_ukernel_4x4c2__sse2);
}

TEST(Q8GEMM_4x4c2__SSE2, k_lt_8_strided_c) {
  TEST_REQUIRES_X86_SSE2;
  GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .np(4)
      .kr(2)
      .m(4)
      .n(4)
      .k(5)
      .cStride(17)
      .test(pytorch_q8gemm_ukernel_4x4c2__sse2);
}

TEST(Q8GEMM_4x4c2__SSE2, k_lt_8_qmin128) {
  TEST_REQUIRES_X86_SSE2;
  GemmMicrokernelTester().mr(4).nr(4).np(4).kr(2).m(4).n(4).k(5).qmin(128).test(
      pytorch_q8gemm_ukernel_4x4c2__sse2);
}

TEST(Q8GEMM_4x4c2__SSE2, k_lt_8_qmax128) {
  TEST_REQUIRES_X86_SSE2;
  GemmMicrokernelTester().mr(4).nr(4).np(4).kr(2).m(4).n(4).k(5).qmax(128).test(
      pytorch_q8gemm_ukernel_4x4c2__sse2);
}

TEST(Q8GEMM_4x4c2__SSE2, k_lt_8_azp0) {
  TEST_REQUIRES_X86_SSE2;
  GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .np(4)
      .kr(2)
      .m(4)
      .n(4)
      .k(5)
      .aZeroPoint(0)
      .test(pytorch_q8gemm_ukernel_4x4c2__sse2);
}

TEST(Q8GEMM_4x4c2__SSE2, k_lt_8_bzp0) {
  TEST_REQUIRES_X86_SSE2;
  GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .np(4)
      .kr(2)
      .m(4)
      .n(4)
      .k(5)
      .bZeroPoint(0)
      .test(pytorch_q8gemm_ukernel_4x4c2__sse2);
}

TEST(Q8GEMM_4x4c2__SSE2, k_lt_8_nozp) {
  TEST_REQUIRES_X86_SSE2;
  GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .np(4)
      .kr(2)
      .m(4)
      .n(4)
      .k(5)
      .aZeroPoint(0)
      .bZeroPoint(0)
      .test(pytorch_q8gemm_ukernel_4x4c2__sse2);
}

TEST(Q8GEMM_4x4c2__SSE2, k_eq_8) {
  TEST_REQUIRES_X86_SSE2;
  GemmMicrokernelTester().mr(4).nr(4).np(4).kr(2).m(4).n(4).k(8).test(
      pytorch_q8gemm_ukernel_4x4c2__sse2);
}

TEST(Q8GEMM_4x4c2__SSE2, k_eq_8_strided_a) {
  TEST_REQUIRES_X86_SSE2;
  GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .np(4)
      .kr(2)
      .m(4)
      .n(4)
      .k(8)
      .aStride(37)
      .test(pytorch_q8gemm_ukernel_4x4c2__sse2);
}

TEST(Q8GEMM_4x4c2__SSE2, k_eq_8_strided_c) {
  TEST_REQUIRES_X86_SSE2;
  GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .np(4)
      .kr(2)
      .m(4)
      .n(4)
      .k(8)
      .cStride(17)
      .test(pytorch_q8gemm_ukernel_4x4c2__sse2);
}

TEST(Q8GEMM_4x4c2__SSE2, k_eq_8_qmin128) {
  TEST_REQUIRES_X86_SSE2;
  GemmMicrokernelTester().mr(4).nr(4).np(4).kr(2).m(4).n(4).k(8).qmin(128).test(
      pytorch_q8gemm_ukernel_4x4c2__sse2);
}

TEST(Q8GEMM_4x4c2__SSE2, k_eq_8_qmax128) {
  TEST_REQUIRES_X86_SSE2;
  GemmMicrokernelTester().mr(4).nr(4).np(4).kr(2).m(4).n(4).k(8).qmax(128).test(
      pytorch_q8gemm_ukernel_4x4c2__sse2);
}

TEST(Q8GEMM_4x4c2__SSE2, k_eq_8_azp0) {
  TEST_REQUIRES_X86_SSE2;
  GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .np(4)
      .kr(2)
      .m(4)
      .n(4)
      .k(8)
      .aZeroPoint(0)
      .test(pytorch_q8gemm_ukernel_4x4c2__sse2);
}

TEST(Q8GEMM_4x4c2__SSE2, k_eq_8_bzp0) {
  TEST_REQUIRES_X86_SSE2;
  GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .np(4)
      .kr(2)
      .m(4)
      .n(4)
      .k(8)
      .bZeroPoint(0)
      .test(pytorch_q8gemm_ukernel_4x4c2__sse2);
}

TEST(Q8GEMM_4x4c2__SSE2, k_eq_8_nozp) {
  TEST_REQUIRES_X86_SSE2;
  GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .np(4)
      .kr(2)
      .m(4)
      .n(4)
      .k(8)
      .aZeroPoint(0)
      .bZeroPoint(0)
      .test(pytorch_q8gemm_ukernel_4x4c2__sse2);
}

TEST(Q8GEMM_4x4c2__SSE2, k_gt_8) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester().mr(4).nr(4).np(4).kr(2).m(4).n(4).k(k).test(
        pytorch_q8gemm_ukernel_4x4c2__sse2);
  }
}

TEST(Q8GEMM_4x4c2__SSE2, k_gt_8_strided_a) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .np(4)
        .kr(2)
        .m(4)
        .n(4)
        .k(k)
        .aStride(37)
        .test(pytorch_q8gemm_ukernel_4x4c2__sse2);
  }
}

TEST(Q8GEMM_4x4c2__SSE2, k_gt_8_strided_c) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .np(4)
        .kr(2)
        .m(4)
        .n(4)
        .k(k)
        .cStride(17)
        .test(pytorch_q8gemm_ukernel_4x4c2__sse2);
  }
}

TEST(Q8GEMM_4x4c2__SSE2, k_gt_8_azp0) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .np(4)
        .kr(2)
        .m(4)
        .n(4)
        .k(k)
        .aZeroPoint(0)
        .test(pytorch_q8gemm_ukernel_4x4c2__sse2);
  }
}

TEST(Q8GEMM_4x4c2__SSE2, k_gt_8_bzp0) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .np(4)
        .kr(2)
        .m(4)
        .n(4)
        .k(k)
        .bZeroPoint(0)
        .test(pytorch_q8gemm_ukernel_4x4c2__sse2);
  }
}

TEST(Q8GEMM_4x4c2__SSE2, k_gt_8_nozp) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t k = 9; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .np(4)
        .kr(2)
        .m(4)
        .n(4)
        .k(k)
        .aZeroPoint(0)
        .bZeroPoint(0)
        .test(pytorch_q8gemm_ukernel_4x4c2__sse2);
  }
}

TEST(Q8GEMM_4x4c2__SSE2, k_gt_8_subtile) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t k = 9; k < 16; k++) {
    for (uint32_t m = 1; m <= 4; m++) {
      for (uint32_t n = 1; n <= 4; n++) {
        GemmMicrokernelTester()
            .mr(4)
            .nr(4)
            .np(4)
            .kr(2)
            .m(m)
            .n(n)
            .k(k)
            .iterations(3)
            .test(pytorch_q8gemm_ukernel_4x4c2__sse2);
      }
    }
  }
}

TEST(Q8GEMM_4x4c2__SSE2, k_div_8) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t k = 16; k < 128; k += 8) {
    GemmMicrokernelTester().mr(4).nr(4).np(4).kr(2).m(4).n(4).k(k).test(
        pytorch_q8gemm_ukernel_4x4c2__sse2);
  }
}

TEST(Q8GEMM_4x4c2__SSE2, k_div_8_strided_a) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t k = 16; k < 128; k += 8) {
    GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .np(4)
        .kr(2)
        .m(4)
        .n(4)
        .k(k)
        .aStride(171)
        .test(pytorch_q8gemm_ukernel_4x4c2__sse2);
  }
}

TEST(Q8GEMM_4x4c2__SSE2, k_div_8_strided_c) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t k = 16; k < 128; k += 8) {
    GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .np(4)
        .kr(2)
        .m(4)
        .n(4)
        .k(k)
        .cStride(17)
        .test(pytorch_q8gemm_ukernel_4x4c2__sse2);
  }
}

TEST(Q8GEMM_4x4c2__SSE2, k_div_8_subtile) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t k = 16; k < 128; k += 24) {
    for (uint32_t m = 1; m <= 4; m++) {
      for (uint32_t n = 1; n <= 4; n++) {
        GemmMicrokernelTester()
            .mr(4)
            .nr(4)
            .np(4)
            .kr(2)
            .m(m)
            .n(n)
            .k(k)
            .iterations(3)
            .test(pytorch_q8gemm_ukernel_4x4c2__sse2);
      }
    }
  }
}
#endif
