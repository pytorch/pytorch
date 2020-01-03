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
#include <qnnpack/q8vadd.h>

#include "vadd-microkernel-tester.h"

#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
TEST(Q8VADD__SSE2, n_eq_8) {
  TEST_REQUIRES_X86_SSE2;
  VAddMicrokernelTester().n(8).test(pytorch_q8vadd_ukernel__sse2);
}

TEST(Q8VADD__SSE2, n_div_8) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 8; n < 128; n += 24) {
    VAddMicrokernelTester().n(n).test(pytorch_q8vadd_ukernel__sse2);
  }
}

TEST(Q8VADD__SSE2, n_gt_8) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 9; n < 16; n++) {
    VAddMicrokernelTester().n(n).test(pytorch_q8vadd_ukernel__sse2);
  }
}

TEST(Q8VADD__SSE2, n_lt_8) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 1; n < 8; n++) {
    VAddMicrokernelTester().n(n).test(pytorch_q8vadd_ukernel__sse2);
  }
}

TEST(Q8VADD__SSE2, inplace_a) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 1; n < 128; n += 11) {
    VAddMicrokernelTester().iterations(1).n(n).inplaceA(true).test(
        pytorch_q8vadd_ukernel__sse2);
  }
}

TEST(Q8VADD__SSE2, inplace_b) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 1; n < 128; n += 11) {
    VAddMicrokernelTester().iterations(1).n(n).inplaceB(true).test(
        pytorch_q8vadd_ukernel__sse2);
  }
}

TEST(Q8VADD__SSE2, inplace_a_and_b) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 1; n < 128; n += 11) {
    VAddMicrokernelTester()
        .iterations(1)
        .n(n)
        .inplaceA(true)
        .inplaceB(true)
        .test(pytorch_q8vadd_ukernel__sse2);
  }
}

TEST(Q8VADD__SSE2, a_scale) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 1; n < 128; n += 11) {
    for (float aScale = 1.0e-2; aScale < 1.0e+2; aScale *= 1.7f) {
      VAddMicrokernelTester().iterations(1).n(n).aScale(aScale).test(
          pytorch_q8vadd_ukernel__sse2);
    }
  }
}

TEST(Q8VADD__SSE2, b_scale) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 1; n < 128; n += 11) {
    for (float bScale = 1.0e-2; bScale < 1.0e+2; bScale *= 1.7f) {
      VAddMicrokernelTester().iterations(1).n(n).bScale(bScale).test(
          pytorch_q8vadd_ukernel__sse2);
    }
  }
}

TEST(Q8VADD__SSE2, y_scale) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 1; n < 128; n += 11) {
    for (float yScale = 1.0e-2; yScale < 1.0e+2; yScale *= 1.7f) {
      VAddMicrokernelTester().iterations(1).n(n).yScale(yScale).test(
          pytorch_q8vadd_ukernel__sse2);
    }
  }
}

TEST(Q8VADD__SSE2, a_zero_point) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 1; n < 128; n += 11) {
    for (int32_t aZeroPoint = 0; aZeroPoint <= 255; aZeroPoint += 51) {
      VAddMicrokernelTester()
          .iterations(1)
          .n(n)
          .aZeroPoint(uint8_t(aZeroPoint))
          .test(pytorch_q8vadd_ukernel__sse2);
    }
  }
}

TEST(Q8VADD__SSE2, b_zero_point) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 1; n < 128; n += 11) {
    for (int32_t bZeroPoint = 0; bZeroPoint <= 255; bZeroPoint += 51) {
      VAddMicrokernelTester()
          .iterations(1)
          .n(n)
          .bZeroPoint(uint8_t(bZeroPoint))
          .test(pytorch_q8vadd_ukernel__sse2);
    }
  }
}

TEST(Q8VADD__SSE2, y_zero_point) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 1; n < 128; n += 11) {
    for (int32_t yZeroPoint = 0; yZeroPoint <= 255; yZeroPoint += 51) {
      VAddMicrokernelTester()
          .iterations(1)
          .n(n)
          .yZeroPoint(uint8_t(yZeroPoint))
          .test(pytorch_q8vadd_ukernel__sse2);
    }
  }
}

TEST(Q8VADD__SSE2, qmin) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 1; n < 128; n += 11) {
    VAddMicrokernelTester().iterations(1).n(n).qmin(128).test(
        pytorch_q8vadd_ukernel__sse2);
  }
}

TEST(Q8VADD__SSE2, qmax) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 1; n < 128; n += 11) {
    VAddMicrokernelTester().iterations(1).n(n).qmax(128).test(
        pytorch_q8vadd_ukernel__sse2);
  }
}
#endif /* CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64 */

#if CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64
TEST(Q8VADD__NEON, n_eq_8) {
  TEST_REQUIRES_ARM_NEON;
  VAddMicrokernelTester().n(8).test(pytorch_q8vadd_ukernel__neon);
}

TEST(Q8VADD__NEON, n_div_8) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 8; n < 128; n += 24) {
    VAddMicrokernelTester().n(n).test(pytorch_q8vadd_ukernel__neon);
  }
}

TEST(Q8VADD__NEON, n_gt_8) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 9; n < 16; n++) {
    VAddMicrokernelTester().n(n).test(pytorch_q8vadd_ukernel__neon);
  }
}

TEST(Q8VADD__NEON, n_lt_8) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 1; n < 8; n++) {
    VAddMicrokernelTester().n(n).test(pytorch_q8vadd_ukernel__neon);
  }
}

TEST(Q8VADD__NEON, inplace_a) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 1; n < 128; n += 11) {
    VAddMicrokernelTester().iterations(1).n(n).inplaceA(true).test(
        pytorch_q8vadd_ukernel__neon);
  }
}

TEST(Q8VADD__NEON, inplace_b) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 1; n < 128; n += 11) {
    VAddMicrokernelTester().iterations(1).n(n).inplaceB(true).test(
        pytorch_q8vadd_ukernel__neon);
  }
}

TEST(Q8VADD__NEON, inplace_a_and_b) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 1; n < 128; n += 11) {
    VAddMicrokernelTester()
        .iterations(1)
        .n(n)
        .inplaceA(true)
        .inplaceB(true)
        .test(pytorch_q8vadd_ukernel__neon);
  }
}

TEST(Q8VADD__NEON, a_scale) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 1; n < 128; n += 11) {
    for (float aScale = 1.0e-2; aScale < 1.0e+2; aScale *= 1.7f) {
      VAddMicrokernelTester().iterations(1).n(n).aScale(aScale).test(
          pytorch_q8vadd_ukernel__neon);
    }
  }
}

TEST(Q8VADD__NEON, b_scale) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 1; n < 128; n += 11) {
    for (float bScale = 1.0e-2; bScale < 1.0e+2; bScale *= 1.7f) {
      VAddMicrokernelTester().iterations(1).n(n).bScale(bScale).test(
          pytorch_q8vadd_ukernel__neon);
    }
  }
}

TEST(Q8VADD__NEON, y_scale) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 1; n < 128; n += 11) {
    for (float yScale = 1.0e-2; yScale < 1.0e+2; yScale *= 1.7f) {
      VAddMicrokernelTester().iterations(1).n(n).yScale(yScale).test(
          pytorch_q8vadd_ukernel__neon);
    }
  }
}

TEST(Q8VADD__NEON, a_zero_point) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 1; n < 128; n += 11) {
    for (int32_t aZeroPoint = 0; aZeroPoint <= 255; aZeroPoint += 51) {
      VAddMicrokernelTester()
          .iterations(1)
          .n(n)
          .aZeroPoint(uint8_t(aZeroPoint))
          .test(pytorch_q8vadd_ukernel__neon);
    }
  }
}

TEST(Q8VADD__NEON, b_zero_point) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 1; n < 128; n += 11) {
    for (int32_t bZeroPoint = 0; bZeroPoint <= 255; bZeroPoint += 51) {
      VAddMicrokernelTester()
          .iterations(1)
          .n(n)
          .bZeroPoint(uint8_t(bZeroPoint))
          .test(pytorch_q8vadd_ukernel__neon);
    }
  }
}

TEST(Q8VADD__NEON, y_zero_point) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 1; n < 128; n += 11) {
    for (int32_t yZeroPoint = 0; yZeroPoint <= 255; yZeroPoint += 51) {
      VAddMicrokernelTester()
          .iterations(1)
          .n(n)
          .yZeroPoint(uint8_t(yZeroPoint))
          .test(pytorch_q8vadd_ukernel__neon);
    }
  }
}

TEST(Q8VADD__NEON, qmin) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 1; n < 128; n += 11) {
    VAddMicrokernelTester().iterations(1).n(n).qmin(128).test(
        pytorch_q8vadd_ukernel__neon);
  }
}

TEST(Q8VADD__NEON, qmax) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 1; n < 128; n += 11) {
    VAddMicrokernelTester().iterations(1).n(n).qmax(128).test(
        pytorch_q8vadd_ukernel__neon);
  }
}
#endif /* CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64 */
