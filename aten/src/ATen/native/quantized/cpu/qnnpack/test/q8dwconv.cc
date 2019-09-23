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
#include <qnnpack/q8dwconv.h>

#include "dwconv-microkernel-tester.h"

#if CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64
TEST(Q8DWCONV_UP8x9__NEON, single_output_channels_eq_8) {
  TEST_REQUIRES_ARM_NEON;
  DWConvMicrokernelTester()
      .kernelHeight(3)
      .kernelWidth(3)
      .cr(8)
      .channels(8)
      .width(1)
      .test(pytorch_q8dwconv_ukernel_up8x9__neon);
}

TEST(Q8DWCONV_UP8x9__NEON, single_output_channels_eq_8_with_qmin) {
  TEST_REQUIRES_ARM_NEON;
  DWConvMicrokernelTester()
      .kernelHeight(3)
      .kernelWidth(3)
      .cr(8)
      .channels(8)
      .width(1)
      .qmin(128)
      .test(pytorch_q8dwconv_ukernel_up8x9__neon);
}

TEST(Q8DWCONV_UP8x9__NEON, single_output_channels_eq_8_with_qmax) {
  TEST_REQUIRES_ARM_NEON;
  DWConvMicrokernelTester()
      .kernelHeight(3)
      .kernelWidth(3)
      .cr(8)
      .channels(8)
      .width(1)
      .qmax(128)
      .test(pytorch_q8dwconv_ukernel_up8x9__neon);
}

TEST(
    Q8DWCONV_UP8x9__NEON,
    single_output_channels_eq_8_with_input_zero_point_only) {
  TEST_REQUIRES_ARM_NEON;
  DWConvMicrokernelTester()
      .kernelHeight(3)
      .kernelWidth(3)
      .cr(8)
      .channels(8)
      .width(1)
      .inputZeroPoint(255)
      .kernelZeroPoint(0)
      .test(pytorch_q8dwconv_ukernel_up8x9__neon);
}

TEST(
    Q8DWCONV_UP8x9__NEON,
    single_output_channels_eq_8_with_kernel_zero_point_only) {
  TEST_REQUIRES_ARM_NEON;
  DWConvMicrokernelTester()
      .kernelHeight(3)
      .kernelWidth(3)
      .cr(8)
      .channels(8)
      .width(1)
      .inputZeroPoint(0)
      .kernelZeroPoint(255)
      .test(pytorch_q8dwconv_ukernel_up8x9__neon);
}

TEST(Q8DWCONV_UP8x9__NEON, multi_output_channels_eq_8) {
  TEST_REQUIRES_ARM_NEON;
  DWConvMicrokernelTester()
      .kernelHeight(3)
      .kernelWidth(3)
      .cr(8)
      .channels(8)
      .width(5)
      .test(pytorch_q8dwconv_ukernel_up8x9__neon);
}

TEST(Q8DWCONV_UP8x9__NEON, multi_output_channels_eq_8_with_subsampling) {
  TEST_REQUIRES_ARM_NEON;
  DWConvMicrokernelTester()
      .kernelHeight(3)
      .kernelWidth(3)
      .subsampling(2)
      .cr(8)
      .channels(8)
      .width(5)
      .test(pytorch_q8dwconv_ukernel_up8x9__neon);
}

TEST(Q8DWCONV_UP8x9__NEON, multi_output_channels_eq_8_with_input_stride) {
  TEST_REQUIRES_ARM_NEON;
  DWConvMicrokernelTester()
      .kernelHeight(3)
      .kernelWidth(3)
      .cr(8)
      .channels(8)
      .width(5)
      .inputStride(17)
      .test(pytorch_q8dwconv_ukernel_up8x9__neon);
}

TEST(Q8DWCONV_UP8x9__NEON, multi_output_channels_eq_8_with_output_stride) {
  TEST_REQUIRES_ARM_NEON;
  DWConvMicrokernelTester()
      .kernelHeight(3)
      .kernelWidth(3)
      .cr(8)
      .channels(8)
      .width(5)
      .outputStride(19)
      .test(pytorch_q8dwconv_ukernel_up8x9__neon);
}

TEST(Q8DWCONV_UP8x9__NEON, single_output_channels_div_8) {
  TEST_REQUIRES_ARM_NEON;
  for (uint32_t channels = 16; channels < 128; channels += 24) {
    DWConvMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(1)
        .test(pytorch_q8dwconv_ukernel_up8x9__neon);
  }
}

TEST(Q8DWCONV_UP8x9__NEON, multi_output_channels_div_8) {
  TEST_REQUIRES_ARM_NEON;
  for (uint32_t channels = 16; channels < 128; channels += 24) {
    DWConvMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(5)
        .test(pytorch_q8dwconv_ukernel_up8x9__neon);
  }
}

TEST(Q8DWCONV_UP8x9__NEON, multi_output_channels_div_8_with_output_stride) {
  TEST_REQUIRES_ARM_NEON;
  for (uint32_t channels = 16; channels < 128; channels += 24) {
    DWConvMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(5)
        .outputStride(171)
        .test(pytorch_q8dwconv_ukernel_up8x9__neon);
  }
}

TEST(Q8DWCONV_UP8x9__NEON, single_output_channels_gt_8) {
  TEST_REQUIRES_ARM_NEON;
  for (uint32_t channels = 9; channels < 16; channels++) {
    DWConvMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(1)
        .test(pytorch_q8dwconv_ukernel_up8x9__neon);
  }
}

TEST(Q8DWCONV_UP8x9__NEON, single_output_channels_gt_8_with_qmin) {
  TEST_REQUIRES_ARM_NEON;
  for (uint32_t channels = 9; channels < 16; channels++) {
    DWConvMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(1)
        .qmin(128)
        .test(pytorch_q8dwconv_ukernel_up8x9__neon);
  }
}

TEST(Q8DWCONV_UP8x9__NEON, single_output_channels_gt_8_with_qmax) {
  TEST_REQUIRES_ARM_NEON;
  for (uint32_t channels = 9; channels < 16; channels++) {
    DWConvMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(1)
        .qmax(128)
        .test(pytorch_q8dwconv_ukernel_up8x9__neon);
  }
}

TEST(
    Q8DWCONV_UP8x9__NEON,
    single_output_channels_gt_8_with_input_zero_point_only) {
  TEST_REQUIRES_ARM_NEON;
  for (uint32_t channels = 9; channels < 16; channels++) {
    DWConvMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(1)
        .inputZeroPoint(255)
        .kernelZeroPoint(0)
        .test(pytorch_q8dwconv_ukernel_up8x9__neon);
  }
}

TEST(
    Q8DWCONV_UP8x9__NEON,
    single_output_channels_gt_8_with_kernel_zero_point_only) {
  TEST_REQUIRES_ARM_NEON;
  for (uint32_t channels = 9; channels < 16; channels++) {
    DWConvMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(1)
        .inputZeroPoint(0)
        .kernelZeroPoint(255)
        .test(pytorch_q8dwconv_ukernel_up8x9__neon);
  }
}

TEST(Q8DWCONV_UP8x9__NEON, multi_output_channels_gt_8) {
  TEST_REQUIRES_ARM_NEON;
  for (uint32_t channels = 9; channels < 16; channels++) {
    DWConvMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(5)
        .test(pytorch_q8dwconv_ukernel_up8x9__neon);
  }
}

TEST(Q8DWCONV_UP8x9__NEON, multi_output_channels_gt_8_with_output_stride) {
  TEST_REQUIRES_ARM_NEON;
  for (uint32_t channels = 9; channels < 16; channels++) {
    DWConvMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(5)
        .outputStride(17)
        .test(pytorch_q8dwconv_ukernel_up8x9__neon);
  }
}

TEST(Q8DWCONV_MP8x25__NEON, single_output_channels_eq_8) {
  TEST_REQUIRES_ARM_NEON;
  DWConvMicrokernelTester()
      .kernelHeight(5)
      .kernelWidth(5)
      .cr(8)
      .channels(8)
      .width(1)
      .test(pytorch_q8dwconv_ukernel_mp8x25__neon);
}

TEST(Q8DWCONV_MP8x25__NEON, multi_output_channels_eq_8_with_subsampling) {
  TEST_REQUIRES_ARM_NEON;
  DWConvMicrokernelTester()
      .kernelHeight(5)
      .kernelWidth(5)
      .subsampling(2)
      .cr(8)
      .channels(8)
      .width(5)
      .test(pytorch_q8dwconv_ukernel_mp8x25__neon);
}

TEST(Q8DWCONV_MP8x25__NEON, multi_output_channels_eq_8_with_input_stride) {
  TEST_REQUIRES_ARM_NEON;
  DWConvMicrokernelTester()
      .kernelHeight(5)
      .kernelWidth(5)
      .cr(8)
      .channels(8)
      .width(5)
      .inputStride(17)
      .test(pytorch_q8dwconv_ukernel_mp8x25__neon);
}

TEST(Q8DWCONV_MP8x25__NEON, multi_output_channels_eq_8_with_output_stride) {
  TEST_REQUIRES_ARM_NEON;
  DWConvMicrokernelTester()
      .kernelHeight(5)
      .kernelWidth(5)
      .cr(8)
      .channels(8)
      .width(5)
      .outputStride(19)
      .test(pytorch_q8dwconv_ukernel_mp8x25__neon);
}

TEST(Q8DWCONV_MP8x25__NEON, single_output_channels_eq_8_with_qmin) {
  TEST_REQUIRES_ARM_NEON;
  DWConvMicrokernelTester()
      .kernelHeight(5)
      .kernelWidth(5)
      .cr(8)
      .channels(8)
      .width(1)
      .qmin(128)
      .test(pytorch_q8dwconv_ukernel_mp8x25__neon);
}

TEST(Q8DWCONV_MP8x25__NEON, single_output_channels_eq_8_with_qmax) {
  TEST_REQUIRES_ARM_NEON;
  DWConvMicrokernelTester()
      .kernelHeight(5)
      .kernelWidth(5)
      .cr(8)
      .channels(8)
      .width(1)
      .qmax(128)
      .test(pytorch_q8dwconv_ukernel_mp8x25__neon);
}

TEST(
    Q8DWCONV_MP8x25__NEON,
    single_output_channels_eq_8_with_input_zero_point_only) {
  TEST_REQUIRES_ARM_NEON;
  DWConvMicrokernelTester()
      .kernelHeight(5)
      .kernelWidth(5)
      .cr(8)
      .channels(8)
      .width(1)
      .inputZeroPoint(255)
      .kernelZeroPoint(0)
      .test(pytorch_q8dwconv_ukernel_mp8x25__neon);
}

TEST(
    Q8DWCONV_MP8x25__NEON,
    single_output_channels_eq_8_with_kernel_zero_point_only) {
  TEST_REQUIRES_ARM_NEON;
  DWConvMicrokernelTester()
      .kernelHeight(5)
      .kernelWidth(5)
      .cr(8)
      .channels(8)
      .width(1)
      .inputZeroPoint(0)
      .kernelZeroPoint(255)
      .test(pytorch_q8dwconv_ukernel_mp8x25__neon);
}

TEST(Q8DWCONV_MP8x25__NEON, multi_output_channels_eq_8) {
  TEST_REQUIRES_ARM_NEON;
  DWConvMicrokernelTester()
      .kernelHeight(5)
      .kernelWidth(5)
      .cr(8)
      .channels(8)
      .width(3)
      .test(pytorch_q8dwconv_ukernel_mp8x25__neon);
}

TEST(Q8DWCONV_MP8x25__NEON, single_output_channels_div_8) {
  TEST_REQUIRES_ARM_NEON;
  for (uint32_t channels = 16; channels < 128; channels += 24) {
    DWConvMicrokernelTester()
        .kernelHeight(5)
        .kernelWidth(5)
        .cr(8)
        .channels(channels)
        .width(1)
        .test(pytorch_q8dwconv_ukernel_mp8x25__neon);
  }
}

TEST(Q8DWCONV_MP8x25__NEON, multi_output_channels_div_8) {
  TEST_REQUIRES_ARM_NEON;
  for (uint32_t channels = 16; channels < 128; channels += 24) {
    DWConvMicrokernelTester()
        .kernelHeight(5)
        .kernelWidth(5)
        .cr(8)
        .channels(channels)
        .width(5)
        .test(pytorch_q8dwconv_ukernel_mp8x25__neon);
  }
}

TEST(Q8DWCONV_MP8x25__NEON, multi_output_channels_div_8_with_output_stride) {
  TEST_REQUIRES_ARM_NEON;
  for (uint32_t channels = 16; channels < 128; channels += 24) {
    DWConvMicrokernelTester()
        .kernelHeight(5)
        .kernelWidth(5)
        .cr(8)
        .channels(channels)
        .width(5)
        .outputStride(171)
        .test(pytorch_q8dwconv_ukernel_mp8x25__neon);
  }
}

TEST(Q8DWCONV_MP8x25__NEON, single_output_channels_gt_8) {
  TEST_REQUIRES_ARM_NEON;
  for (uint32_t channels = 9; channels < 16; channels++) {
    DWConvMicrokernelTester()
        .kernelHeight(5)
        .kernelWidth(5)
        .cr(8)
        .channels(channels)
        .width(1)
        .test(pytorch_q8dwconv_ukernel_mp8x25__neon);
  }
}

TEST(Q8DWCONV_MP8x25__NEON, single_output_channels_gt_8_with_qmin) {
  TEST_REQUIRES_ARM_NEON;
  for (uint32_t channels = 9; channels < 16; channels++) {
    DWConvMicrokernelTester()
        .kernelHeight(5)
        .kernelWidth(5)
        .cr(8)
        .channels(channels)
        .width(1)
        .qmin(128)
        .test(pytorch_q8dwconv_ukernel_mp8x25__neon);
  }
}

TEST(Q8DWCONV_MP8x25__NEON, single_output_channels_gt_8_with_qmax) {
  TEST_REQUIRES_ARM_NEON;
  for (uint32_t channels = 9; channels < 16; channels++) {
    DWConvMicrokernelTester()
        .kernelHeight(5)
        .kernelWidth(5)
        .cr(8)
        .channels(channels)
        .width(1)
        .qmax(128)
        .test(pytorch_q8dwconv_ukernel_mp8x25__neon);
  }
}

TEST(Q8DWCONV_MP8x25__NEON, multi_output_channels_gt_8) {
  TEST_REQUIRES_ARM_NEON;
  for (uint32_t channels = 9; channels < 16; channels++) {
    DWConvMicrokernelTester()
        .kernelHeight(5)
        .kernelWidth(5)
        .cr(8)
        .channels(channels)
        .width(5)
        .test(pytorch_q8dwconv_ukernel_mp8x25__neon);
  }
}

TEST(Q8DWCONV_MP8x25__NEON, multi_output_channels_gt_8_with_output_stride) {
  TEST_REQUIRES_ARM_NEON;
  for (uint32_t channels = 9; channels < 16; channels++) {
    DWConvMicrokernelTester()
        .kernelHeight(5)
        .kernelWidth(5)
        .cr(8)
        .channels(channels)
        .width(5)
        .outputStride(17)
        .test(pytorch_q8dwconv_ukernel_mp8x25__neon);
  }
}
#endif /* CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64 */

#if CPUINFO_ARCH_ARM
TEST(Q8DWCONV_UP8x9__AARCH32_NEON, single_output_channels_eq_8) {
  TEST_REQUIRES_ARM_NEON;
  DWConvMicrokernelTester()
      .kernelHeight(3)
      .kernelWidth(3)
      .cr(8)
      .channels(8)
      .width(1)
      .test(pytorch_q8dwconv_ukernel_up8x9__aarch32_neon);
}

TEST(Q8DWCONV_UP8x9__AARCH32_NEON, single_output_channels_eq_8_with_qmin) {
  TEST_REQUIRES_ARM_NEON;
  DWConvMicrokernelTester()
      .kernelHeight(3)
      .kernelWidth(3)
      .cr(8)
      .channels(8)
      .width(1)
      .qmin(128)
      .test(pytorch_q8dwconv_ukernel_up8x9__aarch32_neon);
}

TEST(Q8DWCONV_UP8x9__AARCH32_NEON, single_output_channels_eq_8_with_qmax) {
  TEST_REQUIRES_ARM_NEON;
  DWConvMicrokernelTester()
      .kernelHeight(3)
      .kernelWidth(3)
      .cr(8)
      .channels(8)
      .width(1)
      .qmax(128)
      .test(pytorch_q8dwconv_ukernel_up8x9__aarch32_neon);
}

TEST(
    Q8DWCONV_UP8x9__AARCH32_NEON,
    single_output_channels_eq_8_with_input_zero_point_only) {
  TEST_REQUIRES_ARM_NEON;
  DWConvMicrokernelTester()
      .kernelHeight(3)
      .kernelWidth(3)
      .cr(8)
      .channels(8)
      .width(1)
      .inputZeroPoint(255)
      .kernelZeroPoint(0)
      .test(pytorch_q8dwconv_ukernel_up8x9__aarch32_neon);
}

TEST(
    Q8DWCONV_UP8x9__AARCH32_NEON,
    single_output_channels_eq_8_with_kernel_zero_point_only) {
  TEST_REQUIRES_ARM_NEON;
  DWConvMicrokernelTester()
      .kernelHeight(3)
      .kernelWidth(3)
      .cr(8)
      .channels(8)
      .width(1)
      .inputZeroPoint(0)
      .kernelZeroPoint(255)
      .test(pytorch_q8dwconv_ukernel_up8x9__aarch32_neon);
}

TEST(Q8DWCONV_UP8x9__AARCH32_NEON, multi_output_channels_eq_8) {
  TEST_REQUIRES_ARM_NEON;
  DWConvMicrokernelTester()
      .kernelHeight(3)
      .kernelWidth(3)
      .cr(8)
      .channels(8)
      .width(5)
      .test(pytorch_q8dwconv_ukernel_up8x9__aarch32_neon);
}

TEST(
    Q8DWCONV_UP8x9__AARCH32_NEON,
    multi_output_channels_eq_8_with_subsampling) {
  TEST_REQUIRES_ARM_NEON;
  DWConvMicrokernelTester()
      .kernelHeight(3)
      .kernelWidth(3)
      .subsampling(2)
      .cr(8)
      .channels(8)
      .width(5)
      .test(pytorch_q8dwconv_ukernel_up8x9__aarch32_neon);
}

TEST(
    Q8DWCONV_UP8x9__AARCH32_NEON,
    multi_output_channels_eq_8_with_input_stride) {
  TEST_REQUIRES_ARM_NEON;
  DWConvMicrokernelTester()
      .kernelHeight(3)
      .kernelWidth(3)
      .cr(8)
      .channels(8)
      .width(5)
      .inputStride(17)
      .test(pytorch_q8dwconv_ukernel_up8x9__aarch32_neon);
}

TEST(
    Q8DWCONV_UP8x9__AARCH32_NEON,
    multi_output_channels_eq_8_with_output_stride) {
  TEST_REQUIRES_ARM_NEON;
  DWConvMicrokernelTester()
      .kernelHeight(3)
      .kernelWidth(3)
      .cr(8)
      .channels(8)
      .width(5)
      .outputStride(19)
      .test(pytorch_q8dwconv_ukernel_up8x9__aarch32_neon);
}

TEST(Q8DWCONV_UP8x9__AARCH32_NEON, single_output_channels_div_8) {
  TEST_REQUIRES_ARM_NEON;
  for (uint32_t channels = 16; channels < 128; channels += 24) {
    DWConvMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(1)
        .test(pytorch_q8dwconv_ukernel_up8x9__aarch32_neon);
  }
}

TEST(Q8DWCONV_UP8x9__AARCH32_NEON, multi_output_channels_div_8) {
  TEST_REQUIRES_ARM_NEON;
  for (uint32_t channels = 16; channels < 128; channels += 24) {
    DWConvMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(5)
        .test(pytorch_q8dwconv_ukernel_up8x9__aarch32_neon);
  }
}

TEST(
    Q8DWCONV_UP8x9__AARCH32_NEON,
    multi_output_channels_div_8_with_output_stride) {
  TEST_REQUIRES_ARM_NEON;
  for (uint32_t channels = 16; channels < 128; channels += 24) {
    DWConvMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(5)
        .outputStride(171)
        .test(pytorch_q8dwconv_ukernel_up8x9__aarch32_neon);
  }
}

TEST(Q8DWCONV_UP8x9__AARCH32_NEON, single_output_channels_gt_8) {
  TEST_REQUIRES_ARM_NEON;
  for (uint32_t channels = 9; channels < 16; channels++) {
    DWConvMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(1)
        .test(pytorch_q8dwconv_ukernel_up8x9__aarch32_neon);
  }
}

TEST(Q8DWCONV_UP8x9__AARCH32_NEON, single_output_channels_gt_8_with_qmin) {
  TEST_REQUIRES_ARM_NEON;
  for (uint32_t channels = 9; channels < 16; channels++) {
    DWConvMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(1)
        .qmin(128)
        .test(pytorch_q8dwconv_ukernel_up8x9__aarch32_neon);
  }
}

TEST(Q8DWCONV_UP8x9__AARCH32_NEON, single_output_channels_gt_8_with_qmax) {
  TEST_REQUIRES_ARM_NEON;
  for (uint32_t channels = 9; channels < 16; channels++) {
    DWConvMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(1)
        .qmax(128)
        .test(pytorch_q8dwconv_ukernel_up8x9__aarch32_neon);
  }
}

TEST(
    Q8DWCONV_UP8x9__AARCH32_NEON,
    single_output_channels_gt_8_with_input_zero_point_only) {
  TEST_REQUIRES_ARM_NEON;
  for (uint32_t channels = 9; channels < 16; channels++) {
    DWConvMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(1)
        .inputZeroPoint(255)
        .kernelZeroPoint(0)
        .test(pytorch_q8dwconv_ukernel_up8x9__aarch32_neon);
  }
}

TEST(
    Q8DWCONV_UP8x9__AARCH32_NEON,
    single_output_channels_gt_8_with_kernel_zero_point_only) {
  TEST_REQUIRES_ARM_NEON;
  for (uint32_t channels = 9; channels < 16; channels++) {
    DWConvMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(1)
        .inputZeroPoint(0)
        .kernelZeroPoint(255)
        .test(pytorch_q8dwconv_ukernel_up8x9__aarch32_neon);
  }
}

TEST(Q8DWCONV_UP8x9__AARCH32_NEON, multi_output_channels_gt_8) {
  TEST_REQUIRES_ARM_NEON;
  for (uint32_t channels = 9; channels < 16; channels++) {
    DWConvMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(5)
        .test(pytorch_q8dwconv_ukernel_up8x9__aarch32_neon);
  }
}

TEST(
    Q8DWCONV_UP8x9__AARCH32_NEON,
    multi_output_channels_gt_8_with_output_stride) {
  TEST_REQUIRES_ARM_NEON;
  for (uint32_t channels = 9; channels < 16; channels++) {
    DWConvMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(5)
        .outputStride(17)
        .test(pytorch_q8dwconv_ukernel_up8x9__aarch32_neon);
  }
}
#endif /* CPUINFO_ARCH_ARM */

#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
TEST(Q8DWCONV_UP8x9__SSE2, single_output_channels_eq_8) {
  TEST_REQUIRES_X86_SSE2;
  DWConvMicrokernelTester()
      .kernelHeight(3)
      .kernelWidth(3)
      .cr(8)
      .channels(8)
      .width(1)
      .test(pytorch_q8dwconv_ukernel_up8x9__sse2);
}

TEST(Q8DWCONV_UP8x9__SSE2, single_output_channels_eq_8_with_qmin) {
  TEST_REQUIRES_X86_SSE2;
  DWConvMicrokernelTester()
      .kernelHeight(3)
      .kernelWidth(3)
      .cr(8)
      .channels(8)
      .width(1)
      .qmin(128)
      .test(pytorch_q8dwconv_ukernel_up8x9__sse2);
}

TEST(Q8DWCONV_UP8x9__SSE2, single_output_channels_eq_8_with_qmax) {
  TEST_REQUIRES_X86_SSE2;
  DWConvMicrokernelTester()
      .kernelHeight(3)
      .kernelWidth(3)
      .cr(8)
      .channels(8)
      .width(1)
      .qmax(128)
      .test(pytorch_q8dwconv_ukernel_up8x9__sse2);
}

TEST(
    Q8DWCONV_UP8x9__SSE2,
    single_output_channels_eq_8_with_input_zero_point_only) {
  TEST_REQUIRES_X86_SSE2;
  DWConvMicrokernelTester()
      .kernelHeight(3)
      .kernelWidth(3)
      .cr(8)
      .channels(8)
      .width(1)
      .inputZeroPoint(255)
      .kernelZeroPoint(0)
      .test(pytorch_q8dwconv_ukernel_up8x9__sse2);
}

TEST(
    Q8DWCONV_UP8x9__SSE2,
    single_output_channels_eq_8_with_kernel_zero_point_only) {
  TEST_REQUIRES_X86_SSE2;
  DWConvMicrokernelTester()
      .kernelHeight(3)
      .kernelWidth(3)
      .cr(8)
      .channels(8)
      .width(1)
      .inputZeroPoint(0)
      .kernelZeroPoint(255)
      .test(pytorch_q8dwconv_ukernel_up8x9__sse2);
}

TEST(Q8DWCONV_UP8x9__SSE2, multi_output_channels_eq_8) {
  TEST_REQUIRES_X86_SSE2;
  DWConvMicrokernelTester()
      .kernelHeight(3)
      .kernelWidth(3)
      .cr(8)
      .channels(8)
      .width(5)
      .test(pytorch_q8dwconv_ukernel_up8x9__sse2);
}

TEST(Q8DWCONV_UP8x9__SSE2, multi_output_channels_eq_8_with_subsampling) {
  TEST_REQUIRES_X86_SSE2;
  DWConvMicrokernelTester()
      .kernelHeight(3)
      .kernelWidth(3)
      .subsampling(2)
      .cr(8)
      .channels(8)
      .width(5)
      .test(pytorch_q8dwconv_ukernel_up8x9__sse2);
}

TEST(Q8DWCONV_UP8x9__SSE2, multi_output_channels_eq_8_with_input_stride) {
  TEST_REQUIRES_X86_SSE2;
  DWConvMicrokernelTester()
      .kernelHeight(3)
      .kernelWidth(3)
      .cr(8)
      .channels(8)
      .width(5)
      .inputStride(17)
      .test(pytorch_q8dwconv_ukernel_up8x9__sse2);
}

TEST(Q8DWCONV_UP8x9__SSE2, multi_output_channels_eq_8_with_output_stride) {
  TEST_REQUIRES_X86_SSE2;
  DWConvMicrokernelTester()
      .kernelHeight(3)
      .kernelWidth(3)
      .cr(8)
      .channels(8)
      .width(5)
      .outputStride(19)
      .test(pytorch_q8dwconv_ukernel_up8x9__sse2);
}

TEST(Q8DWCONV_UP8x9__SSE2, single_output_channels_div_8) {
  TEST_REQUIRES_X86_SSE2;
  for (uint32_t channels = 16; channels < 128; channels += 24) {
    DWConvMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(1)
        .test(pytorch_q8dwconv_ukernel_up8x9__sse2);
  }
}

TEST(Q8DWCONV_UP8x9__SSE2, multi_output_channels_div_8) {
  TEST_REQUIRES_X86_SSE2;
  for (uint32_t channels = 16; channels < 128; channels += 24) {
    DWConvMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(5)
        .test(pytorch_q8dwconv_ukernel_up8x9__sse2);
  }
}

TEST(Q8DWCONV_UP8x9__SSE2, multi_output_channels_div_8_with_output_stride) {
  TEST_REQUIRES_X86_SSE2;
  for (uint32_t channels = 16; channels < 128; channels += 24) {
    DWConvMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(5)
        .outputStride(171)
        .test(pytorch_q8dwconv_ukernel_up8x9__sse2);
  }
}

TEST(Q8DWCONV_UP8x9__SSE2, single_output_channels_gt_8) {
  TEST_REQUIRES_X86_SSE2;
  for (uint32_t channels = 9; channels < 16; channels++) {
    DWConvMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(1)
        .test(pytorch_q8dwconv_ukernel_up8x9__sse2);
  }
}

TEST(Q8DWCONV_UP8x9__SSE2, single_output_channels_gt_8_with_qmin) {
  TEST_REQUIRES_X86_SSE2;
  for (uint32_t channels = 9; channels < 16; channels++) {
    DWConvMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(1)
        .qmin(128)
        .test(pytorch_q8dwconv_ukernel_up8x9__sse2);
  }
}

TEST(Q8DWCONV_UP8x9__SSE2, single_output_channels_gt_8_with_qmax) {
  TEST_REQUIRES_X86_SSE2;
  for (uint32_t channels = 9; channels < 16; channels++) {
    DWConvMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(1)
        .qmax(128)
        .test(pytorch_q8dwconv_ukernel_up8x9__sse2);
  }
}

TEST(
    Q8DWCONV_UP8x9__SSE2,
    single_output_channels_gt_8_with_input_zero_point_only) {
  TEST_REQUIRES_X86_SSE2;
  for (uint32_t channels = 9; channels < 16; channels++) {
    DWConvMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(1)
        .inputZeroPoint(255)
        .kernelZeroPoint(0)
        .test(pytorch_q8dwconv_ukernel_up8x9__sse2);
  }
}

TEST(
    Q8DWCONV_UP8x9__SSE2,
    single_output_channels_gt_8_with_kernel_zero_point_only) {
  TEST_REQUIRES_X86_SSE2;
  for (uint32_t channels = 9; channels < 16; channels++) {
    DWConvMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(1)
        .inputZeroPoint(0)
        .kernelZeroPoint(255)
        .test(pytorch_q8dwconv_ukernel_up8x9__sse2);
  }
}

TEST(Q8DWCONV_UP8x9__SSE2, multi_output_channels_gt_8) {
  TEST_REQUIRES_X86_SSE2;
  for (uint32_t channels = 9; channels < 16; channels++) {
    DWConvMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(5)
        .test(pytorch_q8dwconv_ukernel_up8x9__sse2);
  }
}

TEST(Q8DWCONV_UP8x9__SSE2, multi_output_channels_gt_8_with_output_stride) {
  TEST_REQUIRES_X86_SSE2;
  for (uint32_t channels = 9; channels < 16; channels++) {
    DWConvMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(5)
        .outputStride(17)
        .test(pytorch_q8dwconv_ukernel_up8x9__sse2);
  }
}

TEST(Q8DWCONV_MP8x25__SSE2, single_output_channels_eq_8) {
  TEST_REQUIRES_X86_SSE2;
  DWConvMicrokernelTester()
      .kernelHeight(5)
      .kernelWidth(5)
      .cr(8)
      .channels(8)
      .width(1)
      .test(pytorch_q8dwconv_ukernel_mp8x25__sse2);
}

TEST(Q8DWCONV_MP8x25__SSE2, single_output_channels_eq_8_with_qmin) {
  TEST_REQUIRES_X86_SSE2;
  DWConvMicrokernelTester()
      .kernelHeight(5)
      .kernelWidth(5)
      .cr(8)
      .channels(8)
      .width(1)
      .qmin(128)
      .test(pytorch_q8dwconv_ukernel_mp8x25__sse2);
}

TEST(Q8DWCONV_MP8x25__SSE2, single_output_channels_eq_8_with_qmax) {
  TEST_REQUIRES_X86_SSE2;
  DWConvMicrokernelTester()
      .kernelHeight(5)
      .kernelWidth(5)
      .cr(8)
      .channels(8)
      .width(1)
      .qmax(128)
      .test(pytorch_q8dwconv_ukernel_mp8x25__sse2);
}

TEST(
    Q8DWCONV_MP8x25__SSE2,
    single_output_channels_eq_8_with_input_zero_point_only) {
  TEST_REQUIRES_X86_SSE2;
  DWConvMicrokernelTester()
      .kernelHeight(5)
      .kernelWidth(5)
      .cr(8)
      .channels(8)
      .width(1)
      .inputZeroPoint(255)
      .kernelZeroPoint(0)
      .test(pytorch_q8dwconv_ukernel_mp8x25__sse2);
}

TEST(
    Q8DWCONV_MP8x25__SSE2,
    single_output_channels_eq_8_with_kernel_zero_point_only) {
  TEST_REQUIRES_X86_SSE2;
  DWConvMicrokernelTester()
      .kernelHeight(5)
      .kernelWidth(5)
      .cr(8)
      .channels(8)
      .width(1)
      .inputZeroPoint(0)
      .kernelZeroPoint(255)
      .test(pytorch_q8dwconv_ukernel_mp8x25__sse2);
}

TEST(Q8DWCONV_MP8x25__SSE2, multi_output_channels_eq_8) {
  TEST_REQUIRES_X86_SSE2;
  DWConvMicrokernelTester()
      .kernelHeight(5)
      .kernelWidth(5)
      .cr(8)
      .channels(8)
      .width(5)
      .test(pytorch_q8dwconv_ukernel_mp8x25__sse2);
}

TEST(Q8DWCONV_MP8x25__SSE2, multi_output_channels_eq_8_with_subsampling) {
  TEST_REQUIRES_X86_SSE2;
  DWConvMicrokernelTester()
      .kernelHeight(5)
      .kernelWidth(5)
      .subsampling(2)
      .cr(8)
      .channels(8)
      .width(5)
      .test(pytorch_q8dwconv_ukernel_mp8x25__sse2);
}

TEST(Q8DWCONV_MP8x25__SSE2, multi_output_channels_eq_8_with_input_stride) {
  TEST_REQUIRES_X86_SSE2;
  DWConvMicrokernelTester()
      .kernelHeight(5)
      .kernelWidth(5)
      .cr(8)
      .channels(8)
      .width(5)
      .inputStride(17)
      .test(pytorch_q8dwconv_ukernel_mp8x25__sse2);
}

TEST(Q8DWCONV_MP8x25__SSE2, multi_output_channels_eq_8_with_output_stride) {
  TEST_REQUIRES_X86_SSE2;
  DWConvMicrokernelTester()
      .kernelHeight(5)
      .kernelWidth(5)
      .cr(8)
      .channels(8)
      .width(5)
      .outputStride(19)
      .test(pytorch_q8dwconv_ukernel_mp8x25__sse2);
}

TEST(Q8DWCONV_MP8x25__SSE2, single_output_channels_div_8) {
  TEST_REQUIRES_X86_SSE2;
  for (uint32_t channels = 16; channels < 128; channels += 24) {
    DWConvMicrokernelTester()
        .kernelHeight(5)
        .kernelWidth(5)
        .cr(8)
        .channels(channels)
        .width(1)
        .test(pytorch_q8dwconv_ukernel_mp8x25__sse2);
  }
}

TEST(Q8DWCONV_MP8x25__SSE2, multi_output_channels_div_8) {
  TEST_REQUIRES_X86_SSE2;
  for (uint32_t channels = 16; channels < 128; channels += 24) {
    DWConvMicrokernelTester()
        .kernelHeight(5)
        .kernelWidth(5)
        .cr(8)
        .channels(channels)
        .width(5)
        .test(pytorch_q8dwconv_ukernel_mp8x25__sse2);
  }
}

TEST(Q8DWCONV_MP8x25__SSE2, multi_output_channels_div_8_with_output_stride) {
  TEST_REQUIRES_X86_SSE2;
  for (uint32_t channels = 16; channels < 128; channels += 24) {
    DWConvMicrokernelTester()
        .kernelHeight(5)
        .kernelWidth(5)
        .cr(8)
        .channels(channels)
        .width(5)
        .outputStride(171)
        .test(pytorch_q8dwconv_ukernel_mp8x25__sse2);
  }
}

TEST(Q8DWCONV_MP8x25__SSE2, single_output_channels_gt_8) {
  TEST_REQUIRES_X86_SSE2;
  for (uint32_t channels = 9; channels < 16; channels++) {
    DWConvMicrokernelTester()
        .kernelHeight(5)
        .kernelWidth(5)
        .cr(8)
        .channels(channels)
        .width(1)
        .test(pytorch_q8dwconv_ukernel_mp8x25__sse2);
  }
}

TEST(Q8DWCONV_MP8x25__SSE2, single_output_channels_gt_8_with_qmin) {
  TEST_REQUIRES_X86_SSE2;
  for (uint32_t channels = 9; channels < 16; channels++) {
    DWConvMicrokernelTester()
        .kernelHeight(5)
        .kernelWidth(5)
        .cr(8)
        .channels(channels)
        .width(1)
        .qmin(128)
        .test(pytorch_q8dwconv_ukernel_mp8x25__sse2);
  }
}

TEST(Q8DWCONV_MP8x25__SSE2, single_output_channels_gt_8_with_qmax) {
  TEST_REQUIRES_X86_SSE2;
  for (uint32_t channels = 9; channels < 16; channels++) {
    DWConvMicrokernelTester()
        .kernelHeight(5)
        .kernelWidth(5)
        .cr(8)
        .channels(channels)
        .width(1)
        .qmax(128)
        .test(pytorch_q8dwconv_ukernel_mp8x25__sse2);
  }
}

TEST(
    Q8DWCONV_MP8x25__SSE2,
    single_output_channels_gt_8_with_input_zero_point_only) {
  TEST_REQUIRES_X86_SSE2;
  for (uint32_t channels = 9; channels < 16; channels++) {
    DWConvMicrokernelTester()
        .kernelHeight(5)
        .kernelWidth(5)
        .cr(8)
        .channels(channels)
        .width(1)
        .inputZeroPoint(255)
        .kernelZeroPoint(0)
        .test(pytorch_q8dwconv_ukernel_mp8x25__sse2);
  }
}

TEST(
    Q8DWCONV_MP8x25__SSE2,
    single_output_channels_gt_8_with_kernel_zero_point_only) {
  TEST_REQUIRES_X86_SSE2;
  for (uint32_t channels = 9; channels < 16; channels++) {
    DWConvMicrokernelTester()
        .kernelHeight(5)
        .kernelWidth(5)
        .cr(8)
        .channels(channels)
        .width(1)
        .inputZeroPoint(0)
        .kernelZeroPoint(255)
        .test(pytorch_q8dwconv_ukernel_mp8x25__sse2);
  }
}

TEST(Q8DWCONV_MP8x25__SSE2, multi_output_channels_gt_8) {
  TEST_REQUIRES_X86_SSE2;
  for (uint32_t channels = 9; channels < 16; channels++) {
    DWConvMicrokernelTester()
        .kernelHeight(5)
        .kernelWidth(5)
        .cr(8)
        .channels(channels)
        .width(5)
        .test(pytorch_q8dwconv_ukernel_mp8x25__sse2);
  }
}

TEST(Q8DWCONV_MP8x25__SSE2, multi_output_channels_gt_8_with_output_stride) {
  TEST_REQUIRES_X86_SSE2;
  for (uint32_t channels = 9; channels < 16; channels++) {
    DWConvMicrokernelTester()
        .kernelHeight(5)
        .kernelWidth(5)
        .cr(8)
        .channels(channels)
        .width(5)
        .outputStride(17)
        .test(pytorch_q8dwconv_ukernel_mp8x25__sse2);
  }
}
#endif /* CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64 */
