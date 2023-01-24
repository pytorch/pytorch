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
#include <qnnpack/u8clamp.h>

#include "clamp-microkernel-tester.h"

#if CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64
TEST(U8CLAMP__NEON, n_eq_8) {
  TEST_REQUIRES_ARM_NEON;
  ClampMicrokernelTester().n(8).test(pytorch_u8clamp_ukernel__neon);
}

TEST(U8CLAMP__NEON, n_div_8) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 8; n < 512; n += 8) {
    ClampMicrokernelTester().n(n).test(pytorch_u8clamp_ukernel__neon);
  }
}

TEST(U8CLAMP__NEON, n_gt_8) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 9; n < 16; n++) {
    ClampMicrokernelTester().n(n).test(pytorch_u8clamp_ukernel__neon);
  }
}

TEST(U8CLAMP__NEON, n_lt_8) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 1; n < 8; n++) {
    ClampMicrokernelTester().n(n).test(pytorch_u8clamp_ukernel__neon);
  }
}

TEST(U8CLAMP__NEON, inplace) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 1; n < 128; n += 5) {
    ClampMicrokernelTester().iterations(1).n(n).inplace(true).test(
        pytorch_u8clamp_ukernel__neon);
  }
}

TEST(U8CLAMP__NEON, qmin) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 1; n < 128; n += 11) {
    for (uint8_t qmin = 1; qmin < 255; qmin++) {
      ClampMicrokernelTester().iterations(1).n(n).qmin(qmin).test(
          pytorch_u8clamp_ukernel__neon);
    }
  }
}

TEST(U8CLAMP__NEON, qmax) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 1; n < 128; n += 11) {
    for (uint8_t qmax = 1; qmax < 255; qmax++) {
      ClampMicrokernelTester().iterations(1).n(n).qmax(qmax).test(
          pytorch_u8clamp_ukernel__neon);
    }
  }
}
#endif /* CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64 */

#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
TEST(U8CLAMP__SSE2, n_eq_8) {
  TEST_REQUIRES_X86_SSE2;
  ClampMicrokernelTester().n(8).test(pytorch_u8clamp_ukernel__sse2);
}

TEST(U8CLAMP__SSE2, n_div_8) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 8; n < 512; n += 8) {
    ClampMicrokernelTester().n(n).test(pytorch_u8clamp_ukernel__sse2);
  }
}

TEST(U8CLAMP__SSE2, n_gt_8) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 9; n < 16; n++) {
    ClampMicrokernelTester().n(n).test(pytorch_u8clamp_ukernel__sse2);
  }
}

TEST(U8CLAMP__SSE2, n_lt_8) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 1; n < 8; n++) {
    ClampMicrokernelTester().n(n).test(pytorch_u8clamp_ukernel__sse2);
  }
}

TEST(U8CLAMP__SSE2, inplace) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 1; n < 128; n += 5) {
    ClampMicrokernelTester().iterations(1).n(n).inplace(true).test(
        pytorch_u8clamp_ukernel__sse2);
  }
}

TEST(U8CLAMP__SSE2, qmin) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 1; n < 128; n += 11) {
    for (uint8_t qmin = 1; qmin < 255; qmin++) {
      ClampMicrokernelTester().iterations(1).n(n).qmin(qmin).test(
          pytorch_u8clamp_ukernel__sse2);
    }
  }
}

TEST(U8CLAMP__SSE2, qmax) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 1; n < 128; n += 11) {
    for (uint8_t qmax = 1; qmax < 255; qmax++) {
      ClampMicrokernelTester().iterations(1).n(n).qmax(qmax).test(
          pytorch_u8clamp_ukernel__sse2);
    }
  }
}
#endif /* CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64 */
