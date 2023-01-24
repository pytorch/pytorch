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
#include <qnnpack/sconv.h>

#include "gemm-microkernel-tester.h"

TEST(SCONV_6x8__PSIMD, k_eq_1) {
  GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .np(8)
      .kr(1)
      .m(6)
      .n(8)
      .k(1)
      .aStride(37)
      .test(pytorch_sconv_ukernel_6x8__psimd);
}

TEST(SCONV_6x8__PSIMD, k_eq_1_strided_c) {
  GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .np(8)
      .kr(1)
      .m(6)
      .n(8)
      .k(1)
      .aStride(37)
      .cStride(17)
      .test(pytorch_sconv_ukernel_6x8__psimd);
}

TEST(SCONV_6x8__PSIMD, k_eq_1_qmin128) {
  GemmMicrokernelTester().mr(6).nr(8).np(8).kr(1).m(6).n(8).k(1).qmin(128).test(
      pytorch_sconv_ukernel_6x8__psimd);
}

TEST(SCONV_6x8__PSIMD, k_eq_1_qmax128) {
  GemmMicrokernelTester().mr(6).nr(8).np(8).kr(1).m(6).n(8).k(1).qmax(128).test(
      pytorch_sconv_ukernel_6x8__psimd);
}

TEST(SCONV_6x8__PSIMD, k_gt_1) {
  for (size_t k = 2; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .np(8)
        .kr(1)
        .m(6)
        .n(8)
        .k(k)
        .aStride(37)
        .test(pytorch_sconv_ukernel_6x8__psimd);
  }
}

TEST(SCONV_6x8__PSIMD, k_gt_1_strided_c) {
  for (size_t k = 2; k < 16; k++) {
    GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .np(8)
        .kr(1)
        .m(6)
        .n(8)
        .k(k)
        .aStride(37)
        .cStride(17)
        .test(pytorch_sconv_ukernel_6x8__psimd);
  }
}

TEST(SCONV_6x8__PSIMD, k_gt_1_subtile) {
  for (size_t k = 2; k < 16; k++) {
    for (uint32_t m = 1; m <= 6; m++) {
      for (uint32_t n = 1; n <= 8; n++) {
        GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .np(8)
            .kr(1)
            .m(m)
            .n(n)
            .k(k)
            .aStride(37)
            .iterations(3)
            .test(pytorch_sconv_ukernel_6x8__psimd);
      }
    }
  }
}
