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
#include <qnnpack/q8conv.h>

#include "gemm-microkernel-tester.h"

#if CPUINFO_ARCH_ARM
TEST(Q8CONV_4x8__AARCH32_NEON, k_eq_8) {
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
      .test(pytorch_q8conv_ukernel_4x8__aarch32_neon);
}

TEST(Q8CONV_4x8__AARCH32_NEON, k_eq_8_strided_c) {
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
      .cStride(17)
      .test(pytorch_q8conv_ukernel_4x8__aarch32_neon);
}

TEST(Q8CONV_4x8__AARCH32_NEON, k_eq_8_qmin128) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester().mr(4).nr(8).np(8).kr(1).m(4).n(8).k(8).qmin(128).test(
      pytorch_q8conv_ukernel_4x8__aarch32_neon);
}

TEST(Q8CONV_4x8__AARCH32_NEON, k_eq_8_qmax128) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester().mr(4).nr(8).np(8).kr(1).m(4).n(8).k(8).qmax(128).test(
      pytorch_q8conv_ukernel_4x8__aarch32_neon);
}

TEST(Q8CONV_4x8__AARCH32_NEON, k_eq_8_azp_only) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .np(8)
      .kr(1)
      .m(4)
      .n(8)
      .k(8)
      .aZeroPoint(255)
      .bZeroPoint(0)
      .test(pytorch_q8conv_ukernel_4x8__aarch32_neon);
}

TEST(Q8CONV_4x8__AARCH32_NEON, k_eq_8_bzp_only) {
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
      .bZeroPoint(255)
      .test(pytorch_q8conv_ukernel_4x8__aarch32_neon);
}

TEST(Q8CONV_4x8__AARCH32_NEON, k_gt_8) {
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
        .test(pytorch_q8conv_ukernel_4x8__aarch32_neon);
  }
}

TEST(Q8CONV_4x8__AARCH32_NEON, k_gt_8_strided_c) {
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
        .cStride(17)
        .test(pytorch_q8conv_ukernel_4x8__aarch32_neon);
  }
}

TEST(Q8CONV_4x8__AARCH32_NEON, k_gt_8_azp_only) {
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
        .aZeroPoint(255)
        .bZeroPoint(0)
        .test(pytorch_q8conv_ukernel_4x8__aarch32_neon);
  }
}

TEST(Q8CONV_4x8__AARCH32_NEON, k_gt_8_bzp_only) {
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
        .aZeroPoint(0)
        .bZeroPoint(255)
        .test(pytorch_q8conv_ukernel_4x8__aarch32_neon);
  }
}

TEST(Q8CONV_4x8__AARCH32_NEON, k_gt_8_subtile) {
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
            .aStride(37)
            .iterations(3)
            .test(pytorch_q8conv_ukernel_4x8__aarch32_neon);
      }
    }
  }
}

TEST(Q8CONV_4x8__AARCH32_NEON, k_div_8) {
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
        .test(pytorch_q8conv_ukernel_4x8__aarch32_neon);
  }
}

TEST(Q8CONV_4x8__AARCH32_NEON, k_div_8_strided_c) {
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
        .cStride(17)
        .test(pytorch_q8conv_ukernel_4x8__aarch32_neon);
  }
}

TEST(Q8CONV_4x8__AARCH32_NEON, k_div_8_subtile) {
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
            .aStride(171)
            .iterations(3)
            .test(pytorch_q8conv_ukernel_4x8__aarch32_neon);
      }
    }
  }
}
#endif

#if CPUINFO_ARCH_ARM64
TEST(Q8CONV_8x8__AARCH64_NEON, k_eq_8) {
  GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .np(8)
      .kr(1)
      .m(8)
      .n(8)
      .k(8)
      .aStride(37)
      .test(pytorch_q8conv_ukernel_8x8__aarch64_neon);
}

TEST(Q8CONV_8x8__AARCH64_NEON, k_eq_8_strided_c) {
  GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .np(8)
      .kr(1)
      .m(8)
      .n(8)
      .k(8)
      .aStride(37)
      .cStride(17)
      .test(pytorch_q8conv_ukernel_8x8__aarch64_neon);
}

TEST(Q8CONV_8x8__AARCH64_NEON, k_eq_8_qmin128) {
  GemmMicrokernelTester().mr(8).nr(8).np(8).kr(1).m(8).n(8).k(8).qmin(128).test(
      pytorch_q8conv_ukernel_8x8__aarch64_neon);
}

TEST(Q8CONV_8x8__AARCH64_NEON, k_eq_8_qmax128) {
  GemmMicrokernelTester().mr(8).nr(8).np(8).kr(1).m(8).n(8).k(8).qmax(128).test(
      pytorch_q8conv_ukernel_8x8__aarch64_neon);
}

TEST(Q8CONV_8x8__AARCH64_NEON, k_eq_8_azp_only) {
  GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .np(8)
      .kr(1)
      .m(8)
      .n(8)
      .k(8)
      .aZeroPoint(255)
      .bZeroPoint(0)
      .test(pytorch_q8conv_ukernel_8x8__aarch64_neon);
}

TEST(Q8CONV_8x8__AARCH64_NEON, k_eq_8_bzp_only) {
  GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .np(8)
      .kr(1)
      .m(8)
      .n(8)
      .k(8)
      .aZeroPoint(0)
      .bZeroPoint(255)
      .test(pytorch_q8conv_ukernel_8x8__aarch64_neon);
}

TEST(Q8CONV_8x8__AARCH64_NEON, k_gt_8) {
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
        .test(pytorch_q8conv_ukernel_8x8__aarch64_neon);
  }
}

TEST(Q8CONV_8x8__AARCH64_NEON, k_gt_8_strided_c) {
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
        .cStride(17)
        .test(pytorch_q8conv_ukernel_8x8__aarch64_neon);
  }
}

TEST(Q8CONV_8x8__AARCH64_NEON, k_gt_8_azp_only) {
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
        .aZeroPoint(255)
        .bZeroPoint(0)
        .test(pytorch_q8conv_ukernel_8x8__aarch64_neon);
  }
}

TEST(Q8CONV_8x8__AARCH64_NEON, k_gt_8_bzp_only) {
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
        .aZeroPoint(0)
        .bZeroPoint(255)
        .test(pytorch_q8conv_ukernel_8x8__aarch64_neon);
  }
}

TEST(Q8CONV_8x8__AARCH64_NEON, k_gt_8_subtile) {
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
            .aStride(37)
            .iterations(3)
            .test(pytorch_q8conv_ukernel_8x8__aarch64_neon);
      }
    }
  }
}

TEST(Q8CONV_8x8__AARCH64_NEON, k_div_8) {
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
        .test(pytorch_q8conv_ukernel_8x8__aarch64_neon);
  }
}

TEST(Q8CONV_8x8__AARCH64_NEON, k_div_8_strided_c) {
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
        .cStride(17)
        .test(pytorch_q8conv_ukernel_8x8__aarch64_neon);
  }
}

TEST(Q8CONV_8x8__AARCH64_NEON, k_div_8_subtile) {
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
            .aStride(171)
            .iterations(3)
            .test(pytorch_q8conv_ukernel_8x8__aarch64_neon);
      }
    }
  }
}
#endif

#if CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64
TEST(Q8CONV_4x8__NEON, k_eq_8) {
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
      .test(pytorch_q8conv_ukernel_4x8__neon);
}

TEST(Q8CONV_4x8__NEON, k_eq_8_strided_c) {
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
      .cStride(17)
      .test(pytorch_q8conv_ukernel_4x8__neon);
}

TEST(Q8CONV_4x8__NEON, k_eq_8_qmin128) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester().mr(4).nr(8).np(8).kr(1).m(4).n(8).k(8).qmin(128).test(
      pytorch_q8conv_ukernel_4x8__neon);
}

TEST(Q8CONV_4x8__NEON, k_eq_8_qmax128) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester().mr(4).nr(8).np(8).kr(1).m(4).n(8).k(8).qmax(128).test(
      pytorch_q8conv_ukernel_4x8__neon);
}

TEST(Q8CONV_4x8__NEON, k_eq_8_azp_only) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .np(8)
      .kr(1)
      .m(4)
      .n(8)
      .k(8)
      .aZeroPoint(255)
      .bZeroPoint(0)
      .test(pytorch_q8conv_ukernel_4x8__neon);
}

TEST(Q8CONV_4x8__NEON, k_eq_8_bzp_only) {
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
      .bZeroPoint(255)
      .test(pytorch_q8conv_ukernel_4x8__neon);
}

TEST(Q8CONV_4x8__NEON, k_gt_8) {
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
        .test(pytorch_q8conv_ukernel_4x8__neon);
  }
}

TEST(Q8CONV_4x8__NEON, k_gt_8_strided_c) {
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
        .cStride(17)
        .test(pytorch_q8conv_ukernel_4x8__neon);
  }
}

TEST(Q8CONV_4x8__NEON, k_gt_8_azp_only) {
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
        .aZeroPoint(255)
        .bZeroPoint(0)
        .test(pytorch_q8conv_ukernel_4x8__neon);
  }
}

TEST(Q8CONV_4x8__NEON, k_gt_8_bzp_only) {
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
        .aZeroPoint(0)
        .bZeroPoint(255)
        .test(pytorch_q8conv_ukernel_4x8__neon);
  }
}

TEST(Q8CONV_4x8__NEON, k_gt_8_subtile) {
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
            .aStride(37)
            .iterations(3)
            .test(pytorch_q8conv_ukernel_4x8__neon);
      }
    }
  }
}

TEST(Q8CONV_4x8__NEON, k_div_8) {
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
        .test(pytorch_q8conv_ukernel_4x8__neon);
  }
}

TEST(Q8CONV_4x8__NEON, k_div_8_strided_c) {
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
        .cStride(17)
        .test(pytorch_q8conv_ukernel_4x8__neon);
  }
}

TEST(Q8CONV_4x8__NEON, k_div_8_subtile) {
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
            .aStride(171)
            .iterations(3)
            .test(pytorch_q8conv_ukernel_4x8__neon);
      }
    }
  }
}

TEST(Q8CONV_8x8__NEON, k_eq_8) {
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
      .test(pytorch_q8conv_ukernel_8x8__neon);
}

TEST(Q8CONV_8x8__NEON, k_eq_8_strided_c) {
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
      .cStride(17)
      .test(pytorch_q8conv_ukernel_8x8__neon);
}

TEST(Q8CONV_8x8__NEON, k_eq_8_qmin128) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester().mr(8).nr(8).np(8).kr(1).m(8).n(8).k(8).qmin(128).test(
      pytorch_q8conv_ukernel_8x8__neon);
}

TEST(Q8CONV_8x8__NEON, k_eq_8_qmax128) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester().mr(8).nr(8).np(8).kr(1).m(8).n(8).k(8).qmax(128).test(
      pytorch_q8conv_ukernel_8x8__neon);
}

TEST(Q8CONV_8x8__NEON, k_eq_8_azp_only) {
  TEST_REQUIRES_ARM_NEON;
  GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .np(8)
      .kr(1)
      .m(8)
      .n(8)
      .k(8)
      .aZeroPoint(255)
      .bZeroPoint(0)
      .test(pytorch_q8conv_ukernel_8x8__neon);
}

TEST(Q8CONV_8x8__NEON, k_eq_8_bzp_only) {
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
      .bZeroPoint(255)
      .test(pytorch_q8conv_ukernel_8x8__neon);
}

TEST(Q8CONV_8x8__NEON, k_gt_8) {
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
        .test(pytorch_q8conv_ukernel_8x8__neon);
  }
}

TEST(Q8CONV_8x8__NEON, k_gt_8_strided_c) {
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
        .cStride(17)
        .test(pytorch_q8conv_ukernel_8x8__neon);
  }
}

TEST(Q8CONV_8x8__NEON, k_gt_8_azp_only) {
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
        .aZeroPoint(255)
        .bZeroPoint(0)
        .test(pytorch_q8conv_ukernel_8x8__neon);
  }
}

TEST(Q8CONV_8x8__NEON, k_gt_8_bzp_only) {
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
        .aZeroPoint(0)
        .bZeroPoint(255)
        .test(pytorch_q8conv_ukernel_8x8__neon);
  }
}

TEST(Q8CONV_8x8__NEON, k_gt_8_subtile) {
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
            .aStride(37)
            .iterations(3)
            .test(pytorch_q8conv_ukernel_8x8__neon);
      }
    }
  }
}

TEST(Q8CONV_8x8__NEON, k_div_8) {
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
        .test(pytorch_q8conv_ukernel_8x8__neon);
  }
}

TEST(Q8CONV_8x8__NEON, k_div_8_strided_c) {
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
        .cStride(17)
        .test(pytorch_q8conv_ukernel_8x8__neon);
  }
}

TEST(Q8CONV_8x8__NEON, k_div_8_subtile) {
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
            .aStride(171)
            .iterations(3)
            .test(pytorch_q8conv_ukernel_8x8__neon);
      }
    }
  }
}
#endif

#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
TEST(Q8CONV_4x4c2__SSE2, k_eq_8) {
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
      .test(pytorch_q8conv_ukernel_4x4c2__sse2);
}

TEST(Q8CONV_4x4c2__SSE2, k_eq_8_strided_c) {
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
      .cStride(17)
      .test(pytorch_q8conv_ukernel_4x4c2__sse2);
}

TEST(Q8CONV_4x4c2__SSE2, k_eq_8_qmin128) {
  TEST_REQUIRES_X86_SSE2;
  GemmMicrokernelTester().mr(4).nr(4).np(4).kr(2).m(4).n(4).k(8).qmin(128).test(
      pytorch_q8conv_ukernel_4x4c2__sse2);
}

TEST(Q8CONV_4x4c2__SSE2, k_eq_8_qmax128) {
  TEST_REQUIRES_X86_SSE2;
  GemmMicrokernelTester().mr(4).nr(4).np(4).kr(2).m(4).n(4).k(8).qmax(128).test(
      pytorch_q8conv_ukernel_4x4c2__sse2);
}

TEST(Q8CONV_4x4c2__SSE2, k_eq_8_azp_only) {
  TEST_REQUIRES_X86_SSE2;
  GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .np(4)
      .kr(2)
      .m(4)
      .n(4)
      .k(8)
      .aZeroPoint(255)
      .bZeroPoint(0)
      .test(pytorch_q8conv_ukernel_4x4c2__sse2);
}

TEST(Q8CONV_4x4c2__SSE2, k_eq_8_bzp_only) {
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
      .bZeroPoint(255)
      .test(pytorch_q8conv_ukernel_4x4c2__sse2);
}

TEST(Q8CONV_4x4c2__SSE2, k_gt_8) {
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
        .test(pytorch_q8conv_ukernel_4x4c2__sse2);
  }
}

TEST(Q8CONV_4x4c2__SSE2, k_gt_8_strided_c) {
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
        .cStride(17)
        .test(pytorch_q8conv_ukernel_4x4c2__sse2);
  }
}

TEST(Q8CONV_4x4c2__SSE2, k_gt_8_azp_only) {
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
        .aZeroPoint(255)
        .bZeroPoint(0)
        .test(pytorch_q8conv_ukernel_4x4c2__sse2);
  }
}

TEST(Q8CONV_4x4c2__SSE2, k_gt_8_bzp_only) {
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
        .aZeroPoint(0)
        .bZeroPoint(255)
        .test(pytorch_q8conv_ukernel_4x4c2__sse2);
  }
}

TEST(Q8CONV_4x4c2__SSE2, k_gt_8_subtile) {
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
            .aStride(37)
            .iterations(3)
            .test(pytorch_q8conv_ukernel_4x4c2__sse2);
      }
    }
  }
}

TEST(Q8CONV_4x4c2__SSE2, k_div_8) {
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
        .test(pytorch_q8conv_ukernel_4x4c2__sse2);
  }
}

TEST(Q8CONV_4x4c2__SSE2, k_div_8_strided_c) {
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
        .cStride(17)
        .test(pytorch_q8conv_ukernel_4x4c2__sse2);
  }
}

TEST(Q8CONV_4x4c2__SSE2, k_div_8_subtile) {
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
            .aStride(171)
            .iterations(3)
            .test(pytorch_q8conv_ukernel_4x4c2__sse2);
      }
    }
  }
}
#endif
