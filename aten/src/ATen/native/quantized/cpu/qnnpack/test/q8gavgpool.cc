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
#include <qnnpack/q8gavgpool.h>

#include "gavgpool-microkernel-tester.h"

#if CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64
TEST(Q8GAVGPOOL_UP8x7__NEON, n_eq_8_all_m) {
  TEST_REQUIRES_ARM_NEON;
  GAvgPoolMicrokernelTester().m(7).n(8).test(pytorch_q8gavgpool_ukernel_up8x7__neon);
}

TEST(Q8GAVGPOOL_UP8x7__NEON, n_eq_8_few_m) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t m = 1; m < 7; m++) {
    GAvgPoolMicrokernelTester().m(m).n(8).test(pytorch_q8gavgpool_ukernel_up8x7__neon);
  }
}

TEST(Q8GAVGPOOL_UP8x7__NEON, n_eq_8_all_m_with_x_stride) {
  TEST_REQUIRES_ARM_NEON;
  GAvgPoolMicrokernelTester().m(7).n(8).xStride(11).test(
      pytorch_q8gavgpool_ukernel_up8x7__neon);
}

TEST(Q8GAVGPOOL_UP8x7__NEON, n_eq_8_all_m_with_x_scale) {
  TEST_REQUIRES_ARM_NEON;
  for (float xScale = 0.01f; xScale < 100.0f; xScale *= 3.14159265f) {
    GAvgPoolMicrokernelTester().m(7).n(8).xScale(xScale).test(
        pytorch_q8gavgpool_ukernel_up8x7__neon);
  }
}

TEST(Q8GAVGPOOL_UP8x7__NEON, n_eq_8_all_m_with_x_zero_point) {
  TEST_REQUIRES_ARM_NEON;
  for (int32_t xZeroPoint = 0; xZeroPoint <= 255; xZeroPoint += 51) {
    GAvgPoolMicrokernelTester()
        .m(7)
        .n(8)
        .xZeroPoint(xZeroPoint)
        .test(pytorch_q8gavgpool_ukernel_up8x7__neon);
  }
}

TEST(Q8GAVGPOOL_UP8x7__NEON, n_eq_8_all_m_with_y_scale) {
  TEST_REQUIRES_ARM_NEON;
  for (float yScale = 0.01f; yScale < 100.0f; yScale *= 3.14159265f) {
    GAvgPoolMicrokernelTester().m(7).n(8).yScale(yScale).test(
        pytorch_q8gavgpool_ukernel_up8x7__neon);
  }
}

TEST(Q8GAVGPOOL_UP8x7__NEON, n_eq_8_all_m_with_y_zero_point) {
  TEST_REQUIRES_ARM_NEON;
  for (int32_t yZeroPoint = 0; yZeroPoint <= 255; yZeroPoint += 51) {
    GAvgPoolMicrokernelTester()
        .m(7)
        .n(8)
        .yZeroPoint(yZeroPoint)
        .test(pytorch_q8gavgpool_ukernel_up8x7__neon);
  }
}

TEST(Q8GAVGPOOL_UP8x7__NEON, n_eq_8_all_m_with_y_max) {
  TEST_REQUIRES_ARM_NEON;
  GAvgPoolMicrokernelTester()
      .m(7)
      .n(8)
      .xZeroPoint(128)
      .yZeroPoint(128)
      .xScale(1.0f)
      .yScale(1.0f)
      .yMax(128)
      .test(pytorch_q8gavgpool_ukernel_up8x7__neon);
}

TEST(Q8GAVGPOOL_UP8x7__NEON, n_eq_8_all_m_with_y_min) {
  TEST_REQUIRES_ARM_NEON;
  GAvgPoolMicrokernelTester()
      .m(7)
      .n(8)
      .xZeroPoint(128)
      .yZeroPoint(128)
      .xScale(1.0f)
      .yScale(1.0f)
      .yMin(128)
      .test(pytorch_q8gavgpool_ukernel_up8x7__neon);
}

TEST(Q8GAVGPOOL_UP8x7__NEON, n_div_8_all_m) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 8; n < 128; n += 24) {
    GAvgPoolMicrokernelTester().m(7).n(n).test(pytorch_q8gavgpool_ukernel_up8x7__neon);
  }
}

TEST(Q8GAVGPOOL_UP8x7__NEON, n_div_8_few_m) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 8; n < 128; n += 24) {
    for (size_t m = 1; m < 7; m++) {
      GAvgPoolMicrokernelTester().m(m).n(n).test(
          pytorch_q8gavgpool_ukernel_up8x7__neon);
    }
  }
}

TEST(Q8GAVGPOOL_UP8x7__NEON, n_gt_8_all_m) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 9; n < 16; n++) {
    GAvgPoolMicrokernelTester().m(7).n(n).test(pytorch_q8gavgpool_ukernel_up8x7__neon);
  }
}

TEST(Q8GAVGPOOL_UP8x7__NEON, n_gt_8_few_m) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 9; n < 16; n++) {
    for (size_t m = 1; m < 7; m++) {
      GAvgPoolMicrokernelTester().m(m).n(n).test(
          pytorch_q8gavgpool_ukernel_up8x7__neon);
    }
  }
}

TEST(Q8GAVGPOOL_UP8x7__NEON, n_gt_8_all_m_with_x_scale) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 9; n < 16; n++) {
    for (float xScale = 0.01f; xScale < 100.0f; xScale *= 3.14159265f) {
      GAvgPoolMicrokernelTester().m(7).n(n).xScale(xScale).test(
          pytorch_q8gavgpool_ukernel_up8x7__neon);
    }
  }
}

TEST(Q8GAVGPOOL_UP8x7__NEON, n_gt_8_all_m_with_x_zero_point) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 9; n < 16; n++) {
    for (int32_t xZeroPoint = 0; xZeroPoint <= 255; xZeroPoint += 51) {
      GAvgPoolMicrokernelTester()
          .m(7)
          .n(n)
          .xZeroPoint(xZeroPoint)
          .test(pytorch_q8gavgpool_ukernel_up8x7__neon);
    }
  }
}

TEST(Q8GAVGPOOL_UP8x7__NEON, n_gt_8_all_m_with_y_scale) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 9; n < 16; n++) {
    for (float yScale = 0.01f; yScale < 100.0f; yScale *= 3.14159265f) {
      GAvgPoolMicrokernelTester().m(7).n(n).yScale(yScale).test(
          pytorch_q8gavgpool_ukernel_up8x7__neon);
    }
  }
}

TEST(Q8GAVGPOOL_UP8x7__NEON, n_gt_8_all_m_with_y_zero_point) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 9; n < 16; n++) {
    for (int32_t yZeroPoint = 0; yZeroPoint <= 255; yZeroPoint += 51) {
      GAvgPoolMicrokernelTester()
          .m(7)
          .n(n)
          .yZeroPoint(yZeroPoint)
          .test(pytorch_q8gavgpool_ukernel_up8x7__neon);
    }
  }
}

TEST(Q8GAVGPOOL_UP8x7__NEON, n_gt_8_all_m_with_y_max) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 9; n < 16; n++) {
    GAvgPoolMicrokernelTester()
        .m(7)
        .n(n)
        .xZeroPoint(128)
        .yZeroPoint(128)
        .xScale(1.0f)
        .yScale(1.0f)
        .yMax(128)
        .test(pytorch_q8gavgpool_ukernel_up8x7__neon);
  }
}

TEST(Q8GAVGPOOL_UP8x7__NEON, n_gt_8_all_m_with_y_min) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 9; n < 16; n++) {
    GAvgPoolMicrokernelTester()
        .m(7)
        .n(n)
        .xZeroPoint(128)
        .yZeroPoint(128)
        .xScale(1.0f)
        .yScale(1.0f)
        .yMin(128)
        .test(pytorch_q8gavgpool_ukernel_up8x7__neon);
  }
}

TEST(Q8GAVGPOOL_MP8x7p7q__NEON, n_eq_8_2pass_all_m) {
  TEST_REQUIRES_ARM_NEON;
  GAvgPoolMicrokernelTester().m(14).n(8).nr(8).test(
      pytorch_q8gavgpool_ukernel_mp8x7p7q__neon);
}

TEST(Q8GAVGPOOL_MP8x7p7q__NEON, n_eq_8_2pass_all_m_with_x_stride) {
  TEST_REQUIRES_ARM_NEON;
  GAvgPoolMicrokernelTester().m(14).n(8).nr(8).xStride(11).test(
      pytorch_q8gavgpool_ukernel_mp8x7p7q__neon);
}

TEST(Q8GAVGPOOL_MP8x7p7q__NEON, n_eq_8_2pass_all_m_with_x_scale) {
  TEST_REQUIRES_ARM_NEON;
  for (float xScale = 0.01f; xScale < 100.0f; xScale *= 3.14159265f) {
    GAvgPoolMicrokernelTester().m(14).n(8).nr(8).xScale(xScale).test(
        pytorch_q8gavgpool_ukernel_mp8x7p7q__neon);
  }
}

TEST(Q8GAVGPOOL_MP8x7p7q__NEON, n_eq_8_2pass_all_m_with_x_zero_point) {
  TEST_REQUIRES_ARM_NEON;
  for (int32_t xZeroPoint = 0; xZeroPoint <= 255; xZeroPoint += 51) {
    GAvgPoolMicrokernelTester()
        .m(14)
        .n(8)
        .nr(8)
        .xZeroPoint(xZeroPoint)
        .test(pytorch_q8gavgpool_ukernel_mp8x7p7q__neon);
  }
}

TEST(Q8GAVGPOOL_MP8x7p7q__NEON, n_eq_8_2pass_all_m_with_y_scale) {
  TEST_REQUIRES_ARM_NEON;
  for (float yScale = 0.01f; yScale < 100.0f; yScale *= 3.14159265f) {
    GAvgPoolMicrokernelTester().m(14).n(8).nr(8).yScale(yScale).test(
        pytorch_q8gavgpool_ukernel_mp8x7p7q__neon);
  }
}

TEST(Q8GAVGPOOL_MP8x7p7q__NEON, n_eq_8_2pass_all_m_with_y_zero_point) {
  TEST_REQUIRES_ARM_NEON;
  for (int32_t yZeroPoint = 0; yZeroPoint <= 255; yZeroPoint += 51) {
    GAvgPoolMicrokernelTester()
        .m(14)
        .n(8)
        .nr(8)
        .yZeroPoint(yZeroPoint)
        .test(pytorch_q8gavgpool_ukernel_mp8x7p7q__neon);
  }
}

TEST(Q8GAVGPOOL_MP8x7p7q__NEON, n_eq_8_2pass_all_m_with_y_max) {
  TEST_REQUIRES_ARM_NEON;
  GAvgPoolMicrokernelTester()
      .m(14)
      .n(8)
      .nr(8)
      .xZeroPoint(128)
      .yZeroPoint(128)
      .xScale(1.0f)
      .yScale(1.0f)
      .yMax(128)
      .test(pytorch_q8gavgpool_ukernel_mp8x7p7q__neon);
}

TEST(Q8GAVGPOOL_MP8x7p7q__NEON, n_eq_8_2pass_all_m_with_y_min) {
  TEST_REQUIRES_ARM_NEON;
  GAvgPoolMicrokernelTester()
      .m(14)
      .n(8)
      .nr(8)
      .xZeroPoint(128)
      .yZeroPoint(128)
      .xScale(1.0f)
      .yScale(1.0f)
      .yMin(128)
      .test(pytorch_q8gavgpool_ukernel_mp8x7p7q__neon);
}

TEST(Q8GAVGPOOL_MP8x7p7q__NEON, n_eq_8_2pass_few_m) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t m = 1; m < 7; m++) {
    GAvgPoolMicrokernelTester().m(7 + m).n(8).nr(8).test(
        pytorch_q8gavgpool_ukernel_mp8x7p7q__neon);
  }
}

TEST(Q8GAVGPOOL_MP8x7p7q__NEON, n_eq_8_2pass_few_m_with_x_stride) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t m = 1; m < 7; m++) {
    GAvgPoolMicrokernelTester().m(7 + m).n(8).nr(8).xStride(11).test(
        pytorch_q8gavgpool_ukernel_mp8x7p7q__neon);
  }
}

TEST(Q8GAVGPOOL_MP8x7p7q__NEON, n_eq_8_multipass_all_m) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t m = 14; m <= 35; m += 7) {
    GAvgPoolMicrokernelTester().m(m).n(8).nr(8).test(
        pytorch_q8gavgpool_ukernel_mp8x7p7q__neon);
  }
}

TEST(Q8GAVGPOOL_MP8x7p7q__NEON, n_eq_8_multipass_all_m_with_x_stride) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t m = 14; m <= 35; m += 7) {
    GAvgPoolMicrokernelTester().m(m).n(8).nr(8).test(
        pytorch_q8gavgpool_ukernel_mp8x7p7q__neon);
  }
}

TEST(Q8GAVGPOOL_MP8x7p7q__NEON, n_div_8_2pass_all_m) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 8; n < 128; n += 24) {
    GAvgPoolMicrokernelTester().m(14).n(n).nr(8).test(
        pytorch_q8gavgpool_ukernel_mp8x7p7q__neon);
  }
}

TEST(Q8GAVGPOOL_MP8x7p7q__NEON, n_div_8_2pass_few_m) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 8; n < 128; n += 24) {
    for (size_t m = 1; m < 7; m++) {
      GAvgPoolMicrokernelTester().m(7 + m).n(n).nr(8).test(
          pytorch_q8gavgpool_ukernel_mp8x7p7q__neon);
    }
  }
}

TEST(Q8GAVGPOOL_MP8x7p7q__NEON, n_div_8_multipass_all_m) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 8; n < 128; n += 24) {
    for (size_t m = 14; m <= 35; m += 7) {
      GAvgPoolMicrokernelTester().m(m).n(n).nr(8).nr(8).test(
          pytorch_q8gavgpool_ukernel_mp8x7p7q__neon);
    }
  }
}

TEST(Q8GAVGPOOL_MP8x7p7q__NEON, n_div_8_multipass_all_m_with_x_stride) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 8; n < 128; n += 24) {
    for (size_t m = 14; m <= 35; m += 7) {
      GAvgPoolMicrokernelTester().m(m).n(n).nr(8).nr(8).xStride(131).test(
          pytorch_q8gavgpool_ukernel_mp8x7p7q__neon);
    }
  }
}

TEST(Q8GAVGPOOL_MP8x7p7q__NEON, n_gt_8_2pass_all_m) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 9; n < 16; n++) {
    GAvgPoolMicrokernelTester().m(14).n(n).nr(8).test(
        pytorch_q8gavgpool_ukernel_mp8x7p7q__neon);
  }
}

TEST(Q8GAVGPOOL_MP8x7p7q__NEON, n_gt_8_2pass_all_m_with_x_scale) {
  TEST_REQUIRES_ARM_NEON;
  for (float xScale = 0.01f; xScale < 100.0f; xScale *= 3.14159265f) {
    for (size_t n = 9; n < 16; n++) {
      GAvgPoolMicrokernelTester().m(14).n(n).nr(8).xScale(xScale).test(
          pytorch_q8gavgpool_ukernel_mp8x7p7q__neon);
    }
  }
}

TEST(Q8GAVGPOOL_MP8x7p7q__NEON, n_gt_8_2pass_all_m_with_x_zero_point) {
  TEST_REQUIRES_ARM_NEON;
  for (int32_t xZeroPoint = 0; xZeroPoint <= 255; xZeroPoint += 51) {
    for (size_t n = 9; n < 16; n++) {
      GAvgPoolMicrokernelTester()
          .m(14)
          .n(n)
          .nr(8)
          .xZeroPoint(xZeroPoint)
          .test(pytorch_q8gavgpool_ukernel_mp8x7p7q__neon);
    }
  }
}

TEST(Q8GAVGPOOL_MP8x7p7q__NEON, n_gt_8_2pass_all_m_with_y_scale) {
  TEST_REQUIRES_ARM_NEON;
  for (float yScale = 0.01f; yScale < 100.0f; yScale *= 3.14159265f) {
    for (size_t n = 9; n < 16; n++) {
      GAvgPoolMicrokernelTester().m(14).n(n).nr(8).yScale(yScale).test(
          pytorch_q8gavgpool_ukernel_mp8x7p7q__neon);
    }
  }
}

TEST(Q8GAVGPOOL_MP8x7p7q__NEON, n_gt_8_2pass_all_m_with_y_zero_point) {
  TEST_REQUIRES_ARM_NEON;
  for (int32_t yZeroPoint = 0; yZeroPoint <= 255; yZeroPoint += 51) {
    for (size_t n = 9; n < 16; n++) {
      GAvgPoolMicrokernelTester()
          .m(14)
          .n(n)
          .nr(8)
          .yZeroPoint(yZeroPoint)
          .test(pytorch_q8gavgpool_ukernel_mp8x7p7q__neon);
    }
  }
}

TEST(Q8GAVGPOOL_MP8x7p7q__NEON, n_gt_8_2pass_all_m_with_y_max) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 9; n < 16; n++) {
    GAvgPoolMicrokernelTester()
        .m(14)
        .n(n)
        .nr(8)
        .xZeroPoint(128)
        .yZeroPoint(128)
        .xScale(1.0f)
        .yScale(1.0f)
        .yMax(128)
        .test(pytorch_q8gavgpool_ukernel_mp8x7p7q__neon);
  }
}

TEST(Q8GAVGPOOL_MP8x7p7q__NEON, n_gt_8_2pass_all_m_with_y_min) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 9; n < 16; n++) {
    GAvgPoolMicrokernelTester()
        .m(14)
        .n(n)
        .nr(8)
        .xZeroPoint(128)
        .yZeroPoint(128)
        .xScale(1.0f)
        .yScale(1.0f)
        .yMin(128)
        .test(pytorch_q8gavgpool_ukernel_mp8x7p7q__neon);
  }
}

TEST(Q8GAVGPOOL_MP8x7p7q__NEON, n_gt_8_2pass_few_m) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 9; n < 16; n++) {
    for (size_t m = 1; m < 7; m++) {
      GAvgPoolMicrokernelTester().m(7 + m).n(n).nr(8).test(
          pytorch_q8gavgpool_ukernel_mp8x7p7q__neon);
    }
  }
}

TEST(Q8GAVGPOOL_MP8x7p7q__NEON, n_gt_8_multipass_all_m) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 9; n < 16; n++) {
    for (size_t m = 14; m <= 35; m += 7) {
      GAvgPoolMicrokernelTester().m(m).n(n).nr(8).test(
          pytorch_q8gavgpool_ukernel_mp8x7p7q__neon);
    }
  }
}

TEST(Q8GAVGPOOL_MP8x7p7q__NEON, n_gt_8_multipass_all_m_with_x_stride) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 9; n < 16; n++) {
    for (size_t m = 14; m <= 35; m += 7) {
      GAvgPoolMicrokernelTester().m(m).n(n).nr(8).xStride(23).test(
          pytorch_q8gavgpool_ukernel_mp8x7p7q__neon);
    }
  }
}

TEST(Q8GAVGPOOL_UP8xM__NEON, n_lt_8_small_m) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 1; n < 8; n++) {
    for (size_t m = 1; m < 8; m++) {
      GAvgPoolMicrokernelTester().m(m).n(n).test(
          pytorch_q8gavgpool_ukernel_up8xm__neon);
    }
  }
}

TEST(Q8GAVGPOOL_UP8xM__NEON, n_lt_8_large_m) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 1; n < 8; n++) {
    for (size_t m = 8; m < 16; m++) {
      GAvgPoolMicrokernelTester().m(m).n(n).test(
          pytorch_q8gavgpool_ukernel_up8xm__neon);
    }
  }
}

TEST(Q8GAVGPOOL_UP8xM__NEON, n_lt_8_with_x_scale) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 1; n < 8; n++) {
    for (size_t m = 1; m < 16; m += 5) {
      for (float xScale = 0.01f; xScale < 100.0f; xScale *= 3.14159265f) {
        GAvgPoolMicrokernelTester().m(m).n(n).xScale(xScale).test(
            pytorch_q8gavgpool_ukernel_up8xm__neon);
      }
    }
  }
}

TEST(Q8GAVGPOOL_UP8xM__NEON, n_lt_8_with_x_zero_point) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 1; n < 8; n++) {
    for (size_t m = 1; m < 16; m += 5) {
      for (int32_t xZeroPoint = 0; xZeroPoint <= 255; xZeroPoint += 51) {
        GAvgPoolMicrokernelTester()
            .m(m)
            .n(n)
            .xZeroPoint(xZeroPoint)
            .test(pytorch_q8gavgpool_ukernel_up8xm__neon);
      }
    }
  }
}

TEST(Q8GAVGPOOL_UP8xM__NEON, n_lt_8_with_y_scale) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 1; n < 8; n++) {
    for (size_t m = 1; m < 16; m += 5) {
      for (float yScale = 0.01f; yScale < 100.0f; yScale *= 3.14159265f) {
        GAvgPoolMicrokernelTester().m(m).n(n).yScale(yScale).test(
            pytorch_q8gavgpool_ukernel_up8xm__neon);
      }
    }
  }
}

TEST(Q8GAVGPOOL_UP8xM__NEON, n_lt_8_with_y_zero_point) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 1; n < 8; n++) {
    for (size_t m = 1; m < 16; m += 5) {
      for (int32_t yZeroPoint = 0; yZeroPoint <= 255; yZeroPoint += 51) {
        GAvgPoolMicrokernelTester()
            .m(m)
            .n(n)
            .yZeroPoint(yZeroPoint)
            .test(pytorch_q8gavgpool_ukernel_up8xm__neon);
      }
    }
  }
}

TEST(Q8GAVGPOOL_UP8xM__NEON, n_lt_8_with_y_max) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 1; n < 8; n++) {
    for (size_t m = 1; m < 16; m += 5) {
      GAvgPoolMicrokernelTester()
          .m(m)
          .n(n)
          .xZeroPoint(128)
          .yZeroPoint(128)
          .xScale(1.0f)
          .yScale(1.0f)
          .yMax(128)
          .test(pytorch_q8gavgpool_ukernel_up8xm__neon);
    }
  }
}

TEST(Q8GAVGPOOL_UP8xM__NEON, n_lt_8_with_y_min) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 1; n < 8; n++) {
    for (size_t m = 1; m < 16; m += 5) {
      GAvgPoolMicrokernelTester()
          .m(m)
          .n(n)
          .xZeroPoint(128)
          .yZeroPoint(128)
          .xScale(1.0f)
          .yScale(1.0f)
          .yMin(128)
          .test(pytorch_q8gavgpool_ukernel_up8xm__neon);
    }
  }
}
#endif /* CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64 */

#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
TEST(Q8GAVGPOOL_UP8x7__SSE2, n_eq_8_all_m) {
  TEST_REQUIRES_X86_SSE2;
  GAvgPoolMicrokernelTester().m(7).n(8).test(pytorch_q8gavgpool_ukernel_up8x7__sse2);
}

TEST(Q8GAVGPOOL_UP8x7__SSE2, n_eq_8_few_m) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t m = 1; m < 7; m++) {
    GAvgPoolMicrokernelTester().m(m).n(8).test(pytorch_q8gavgpool_ukernel_up8x7__sse2);
  }
}

TEST(Q8GAVGPOOL_UP8x7__SSE2, n_eq_8_all_m_with_x_stride) {
  TEST_REQUIRES_X86_SSE2;
  GAvgPoolMicrokernelTester().m(7).n(8).xStride(11).test(
      pytorch_q8gavgpool_ukernel_up8x7__sse2);
}

TEST(Q8GAVGPOOL_UP8x7__SSE2, n_eq_8_all_m_with_x_scale) {
  TEST_REQUIRES_X86_SSE2;
  for (float xScale = 0.01f; xScale < 100.0f; xScale *= 3.14159265f) {
    GAvgPoolMicrokernelTester().m(7).n(8).xScale(xScale).test(
        pytorch_q8gavgpool_ukernel_up8x7__sse2);
  }
}

TEST(Q8GAVGPOOL_UP8x7__SSE2, n_eq_8_all_m_with_x_zero_point) {
  TEST_REQUIRES_X86_SSE2;
  for (int32_t xZeroPoint = 0; xZeroPoint <= 255; xZeroPoint += 51) {
    GAvgPoolMicrokernelTester()
        .m(7)
        .n(8)
        .xZeroPoint(xZeroPoint)
        .test(pytorch_q8gavgpool_ukernel_up8x7__sse2);
  }
}

TEST(Q8GAVGPOOL_UP8x7__SSE2, n_eq_8_all_m_with_y_scale) {
  TEST_REQUIRES_X86_SSE2;
  for (float yScale = 0.01f; yScale < 100.0f; yScale *= 3.14159265f) {
    GAvgPoolMicrokernelTester().m(7).n(8).yScale(yScale).test(
        pytorch_q8gavgpool_ukernel_up8x7__sse2);
  }
}

TEST(Q8GAVGPOOL_UP8x7__SSE2, n_eq_8_all_m_with_y_zero_point) {
  TEST_REQUIRES_X86_SSE2;
  for (int32_t yZeroPoint = 0; yZeroPoint <= 255; yZeroPoint += 51) {
    GAvgPoolMicrokernelTester()
        .m(7)
        .n(8)
        .yZeroPoint(yZeroPoint)
        .test(pytorch_q8gavgpool_ukernel_up8x7__sse2);
  }
}

TEST(Q8GAVGPOOL_UP8x7__SSE2, n_eq_8_all_m_with_y_max) {
  TEST_REQUIRES_X86_SSE2;
  GAvgPoolMicrokernelTester()
      .m(7)
      .n(8)
      .xZeroPoint(128)
      .yZeroPoint(128)
      .xScale(1.0f)
      .yScale(1.0f)
      .yMax(128)
      .test(pytorch_q8gavgpool_ukernel_up8x7__sse2);
}

TEST(Q8GAVGPOOL_UP8x7__SSE2, n_eq_8_all_m_with_y_min) {
  TEST_REQUIRES_X86_SSE2;
  GAvgPoolMicrokernelTester()
      .m(7)
      .n(8)
      .xZeroPoint(128)
      .yZeroPoint(128)
      .xScale(1.0f)
      .yScale(1.0f)
      .yMin(128)
      .test(pytorch_q8gavgpool_ukernel_up8x7__sse2);
}

TEST(Q8GAVGPOOL_UP8x7__SSE2, n_div_8_all_m) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 8; n < 128; n += 24) {
    GAvgPoolMicrokernelTester().m(7).n(n).test(pytorch_q8gavgpool_ukernel_up8x7__sse2);
  }
}

TEST(Q8GAVGPOOL_UP8x7__SSE2, n_div_8_few_m) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 8; n < 128; n += 24) {
    for (size_t m = 1; m < 7; m++) {
      GAvgPoolMicrokernelTester().m(m).n(n).test(
          pytorch_q8gavgpool_ukernel_up8x7__sse2);
    }
  }
}

TEST(Q8GAVGPOOL_UP8x7__SSE2, n_gt_8_all_m) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 9; n < 16; n++) {
    GAvgPoolMicrokernelTester().m(7).n(n).test(pytorch_q8gavgpool_ukernel_up8x7__sse2);
  }
}

TEST(Q8GAVGPOOL_UP8x7__SSE2, n_gt_8_few_m) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 9; n < 16; n++) {
    for (size_t m = 1; m < 7; m++) {
      GAvgPoolMicrokernelTester().m(m).n(n).test(
          pytorch_q8gavgpool_ukernel_up8x7__sse2);
    }
  }
}

TEST(Q8GAVGPOOL_UP8x7__SSE2, n_gt_8_all_m_with_x_scale) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 9; n < 16; n++) {
    for (float xScale = 0.01f; xScale < 100.0f; xScale *= 3.14159265f) {
      GAvgPoolMicrokernelTester().m(7).n(n).xScale(xScale).test(
          pytorch_q8gavgpool_ukernel_up8x7__sse2);
    }
  }
}

TEST(Q8GAVGPOOL_UP8x7__SSE2, n_gt_8_all_m_with_x_zero_point) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 9; n < 16; n++) {
    for (int32_t xZeroPoint = 0; xZeroPoint <= 255; xZeroPoint += 51) {
      GAvgPoolMicrokernelTester()
          .m(7)
          .n(n)
          .xZeroPoint(xZeroPoint)
          .test(pytorch_q8gavgpool_ukernel_up8x7__sse2);
    }
  }
}

TEST(Q8GAVGPOOL_UP8x7__SSE2, n_gt_8_all_m_with_y_scale) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 9; n < 16; n++) {
    for (float yScale = 0.01f; yScale < 100.0f; yScale *= 3.14159265f) {
      GAvgPoolMicrokernelTester().m(7).n(n).yScale(yScale).test(
          pytorch_q8gavgpool_ukernel_up8x7__sse2);
    }
  }
}

TEST(Q8GAVGPOOL_UP8x7__SSE2, n_gt_8_all_m_with_y_zero_point) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 9; n < 16; n++) {
    for (int32_t yZeroPoint = 0; yZeroPoint <= 255; yZeroPoint += 51) {
      GAvgPoolMicrokernelTester()
          .m(7)
          .n(n)
          .yZeroPoint(yZeroPoint)
          .test(pytorch_q8gavgpool_ukernel_up8x7__sse2);
    }
  }
}

TEST(Q8GAVGPOOL_UP8x7__SSE2, n_gt_8_all_m_with_y_max) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 9; n < 16; n++) {
    GAvgPoolMicrokernelTester()
        .m(7)
        .n(n)
        .xZeroPoint(128)
        .yZeroPoint(128)
        .xScale(1.0f)
        .yScale(1.0f)
        .yMax(128)
        .test(pytorch_q8gavgpool_ukernel_up8x7__sse2);
  }
}

TEST(Q8GAVGPOOL_UP8x7__SSE2, n_gt_8_all_m_with_y_min) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 9; n < 16; n++) {
    GAvgPoolMicrokernelTester()
        .m(7)
        .n(n)
        .xZeroPoint(128)
        .yZeroPoint(128)
        .xScale(1.0f)
        .yScale(1.0f)
        .yMin(128)
        .test(pytorch_q8gavgpool_ukernel_up8x7__sse2);
  }
}

TEST(Q8GAVGPOOL_MP8x7p7q__SSE2, n_eq_8_2pass_all_m) {
  TEST_REQUIRES_X86_SSE2;
  GAvgPoolMicrokernelTester().m(14).n(8).nr(8).test(
      pytorch_q8gavgpool_ukernel_mp8x7p7q__sse2);
}

TEST(Q8GAVGPOOL_MP8x7p7q__SSE2, n_eq_8_2pass_all_m_with_x_stride) {
  TEST_REQUIRES_X86_SSE2;
  GAvgPoolMicrokernelTester().m(14).n(8).nr(8).xStride(11).test(
      pytorch_q8gavgpool_ukernel_mp8x7p7q__sse2);
}

TEST(Q8GAVGPOOL_MP8x7p7q__SSE2, n_eq_8_2pass_all_m_with_x_scale) {
  TEST_REQUIRES_X86_SSE2;
  for (float xScale = 0.01f; xScale < 100.0f; xScale *= 3.14159265f) {
    GAvgPoolMicrokernelTester().m(14).n(8).nr(8).xScale(xScale).test(
        pytorch_q8gavgpool_ukernel_mp8x7p7q__sse2);
  }
}

TEST(Q8GAVGPOOL_MP8x7p7q__SSE2, n_eq_8_2pass_all_m_with_x_zero_point) {
  TEST_REQUIRES_X86_SSE2;
  for (int32_t xZeroPoint = 0; xZeroPoint <= 255; xZeroPoint += 51) {
    GAvgPoolMicrokernelTester()
        .m(14)
        .n(8)
        .nr(8)
        .xZeroPoint(xZeroPoint)
        .test(pytorch_q8gavgpool_ukernel_mp8x7p7q__sse2);
  }
}

TEST(Q8GAVGPOOL_MP8x7p7q__SSE2, n_eq_8_2pass_all_m_with_y_scale) {
  TEST_REQUIRES_X86_SSE2;
  for (float yScale = 0.01f; yScale < 100.0f; yScale *= 3.14159265f) {
    GAvgPoolMicrokernelTester().m(14).n(8).nr(8).yScale(yScale).test(
        pytorch_q8gavgpool_ukernel_mp8x7p7q__sse2);
  }
}

TEST(Q8GAVGPOOL_MP8x7p7q__SSE2, n_eq_8_2pass_all_m_with_y_zero_point) {
  TEST_REQUIRES_X86_SSE2;
  for (int32_t yZeroPoint = 0; yZeroPoint <= 255; yZeroPoint += 51) {
    GAvgPoolMicrokernelTester()
        .m(14)
        .n(8)
        .nr(8)
        .yZeroPoint(yZeroPoint)
        .test(pytorch_q8gavgpool_ukernel_mp8x7p7q__sse2);
  }
}

TEST(Q8GAVGPOOL_MP8x7p7q__SSE2, n_eq_8_2pass_all_m_with_y_max) {
  TEST_REQUIRES_X86_SSE2;
  GAvgPoolMicrokernelTester()
      .m(14)
      .n(8)
      .nr(8)
      .xZeroPoint(128)
      .yZeroPoint(128)
      .xScale(1.0f)
      .yScale(1.0f)
      .yMax(128)
      .test(pytorch_q8gavgpool_ukernel_mp8x7p7q__sse2);
}

TEST(Q8GAVGPOOL_MP8x7p7q__SSE2, n_eq_8_2pass_all_m_with_y_min) {
  TEST_REQUIRES_X86_SSE2;
  GAvgPoolMicrokernelTester()
      .m(14)
      .n(8)
      .nr(8)
      .xZeroPoint(128)
      .yZeroPoint(128)
      .xScale(1.0f)
      .yScale(1.0f)
      .yMin(128)
      .test(pytorch_q8gavgpool_ukernel_mp8x7p7q__sse2);
}

TEST(Q8GAVGPOOL_MP8x7p7q__SSE2, n_eq_8_2pass_few_m) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t m = 1; m < 7; m++) {
    GAvgPoolMicrokernelTester().m(7 + m).n(8).nr(8).test(
        pytorch_q8gavgpool_ukernel_mp8x7p7q__sse2);
  }
}

TEST(Q8GAVGPOOL_MP8x7p7q__SSE2, n_eq_8_2pass_few_m_with_x_stride) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t m = 1; m < 7; m++) {
    GAvgPoolMicrokernelTester().m(7 + m).n(8).nr(8).xStride(11).test(
        pytorch_q8gavgpool_ukernel_mp8x7p7q__sse2);
  }
}

TEST(Q8GAVGPOOL_MP8x7p7q__SSE2, n_eq_8_multipass_all_m) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t m = 14; m <= 35; m += 7) {
    GAvgPoolMicrokernelTester().m(m).n(8).nr(8).test(
        pytorch_q8gavgpool_ukernel_mp8x7p7q__sse2);
  }
}

TEST(Q8GAVGPOOL_MP8x7p7q__SSE2, n_eq_8_multipass_all_m_with_x_stride) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t m = 14; m <= 35; m += 7) {
    GAvgPoolMicrokernelTester().m(m).n(8).nr(8).test(
        pytorch_q8gavgpool_ukernel_mp8x7p7q__sse2);
  }
}

TEST(Q8GAVGPOOL_MP8x7p7q__SSE2, n_div_8_2pass_all_m) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 8; n < 128; n += 24) {
    GAvgPoolMicrokernelTester().m(14).n(n).nr(8).test(
        pytorch_q8gavgpool_ukernel_mp8x7p7q__sse2);
  }
}

TEST(Q8GAVGPOOL_MP8x7p7q__SSE2, n_div_8_2pass_few_m) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 8; n < 128; n += 24) {
    for (size_t m = 1; m < 7; m++) {
      GAvgPoolMicrokernelTester().m(7 + m).n(n).nr(8).test(
          pytorch_q8gavgpool_ukernel_mp8x7p7q__sse2);
    }
  }
}

TEST(Q8GAVGPOOL_MP8x7p7q__SSE2, n_div_8_multipass_all_m) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 8; n < 128; n += 24) {
    for (size_t m = 14; m <= 35; m += 7) {
      GAvgPoolMicrokernelTester().m(m).n(n).nr(8).nr(8).test(
          pytorch_q8gavgpool_ukernel_mp8x7p7q__sse2);
    }
  }
}

TEST(Q8GAVGPOOL_MP8x7p7q__SSE2, n_div_8_multipass_all_m_with_x_stride) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 8; n < 128; n += 24) {
    for (size_t m = 14; m <= 35; m += 7) {
      GAvgPoolMicrokernelTester().m(m).n(n).nr(8).nr(8).xStride(131).test(
          pytorch_q8gavgpool_ukernel_mp8x7p7q__sse2);
    }
  }
}

TEST(Q8GAVGPOOL_MP8x7p7q__SSE2, n_gt_8_2pass_all_m) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 9; n < 16; n++) {
    GAvgPoolMicrokernelTester().m(14).n(n).nr(8).test(
        pytorch_q8gavgpool_ukernel_mp8x7p7q__sse2);
  }
}

TEST(Q8GAVGPOOL_MP8x7p7q__SSE2, n_gt_8_2pass_all_m_with_x_scale) {
  TEST_REQUIRES_X86_SSE2;
  for (float xScale = 0.01f; xScale < 100.0f; xScale *= 3.14159265f) {
    for (size_t n = 9; n < 16; n++) {
      GAvgPoolMicrokernelTester().m(14).n(n).nr(8).xScale(xScale).test(
          pytorch_q8gavgpool_ukernel_mp8x7p7q__sse2);
    }
  }
}

TEST(Q8GAVGPOOL_MP8x7p7q__SSE2, n_gt_8_2pass_all_m_with_x_zero_point) {
  TEST_REQUIRES_X86_SSE2;
  for (int32_t xZeroPoint = 0; xZeroPoint <= 255; xZeroPoint += 51) {
    for (size_t n = 9; n < 16; n++) {
      GAvgPoolMicrokernelTester()
          .m(14)
          .n(n)
          .nr(8)
          .xZeroPoint(xZeroPoint)
          .test(pytorch_q8gavgpool_ukernel_mp8x7p7q__sse2);
    }
  }
}

TEST(Q8GAVGPOOL_MP8x7p7q__SSE2, n_gt_8_2pass_all_m_with_y_scale) {
  TEST_REQUIRES_X86_SSE2;
  for (float yScale = 0.01f; yScale < 100.0f; yScale *= 3.14159265f) {
    for (size_t n = 9; n < 16; n++) {
      GAvgPoolMicrokernelTester().m(14).n(n).nr(8).yScale(yScale).test(
          pytorch_q8gavgpool_ukernel_mp8x7p7q__sse2);
    }
  }
}

TEST(Q8GAVGPOOL_MP8x7p7q__SSE2, n_gt_8_2pass_all_m_with_y_zero_point) {
  TEST_REQUIRES_X86_SSE2;
  for (int32_t yZeroPoint = 0; yZeroPoint <= 255; yZeroPoint += 51) {
    for (size_t n = 9; n < 16; n++) {
      GAvgPoolMicrokernelTester()
          .m(14)
          .n(n)
          .nr(8)
          .yZeroPoint(yZeroPoint)
          .test(pytorch_q8gavgpool_ukernel_mp8x7p7q__sse2);
    }
  }
}

TEST(Q8GAVGPOOL_MP8x7p7q__SSE2, n_gt_8_2pass_all_m_with_y_max) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 9; n < 16; n++) {
    GAvgPoolMicrokernelTester()
        .m(14)
        .n(n)
        .nr(8)
        .xZeroPoint(128)
        .yZeroPoint(128)
        .xScale(1.0f)
        .yScale(1.0f)
        .yMax(128)
        .test(pytorch_q8gavgpool_ukernel_mp8x7p7q__sse2);
  }
}

TEST(Q8GAVGPOOL_MP8x7p7q__SSE2, n_gt_8_2pass_all_m_with_y_min) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 9; n < 16; n++) {
    GAvgPoolMicrokernelTester()
        .m(14)
        .n(n)
        .nr(8)
        .xZeroPoint(128)
        .yZeroPoint(128)
        .xScale(1.0f)
        .yScale(1.0f)
        .yMin(128)
        .test(pytorch_q8gavgpool_ukernel_mp8x7p7q__sse2);
  }
}

TEST(Q8GAVGPOOL_MP8x7p7q__SSE2, n_gt_8_2pass_few_m) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 9; n < 16; n++) {
    for (size_t m = 1; m < 7; m++) {
      GAvgPoolMicrokernelTester().m(7 + m).n(n).nr(8).test(
          pytorch_q8gavgpool_ukernel_mp8x7p7q__sse2);
    }
  }
}

TEST(Q8GAVGPOOL_MP8x7p7q__SSE2, n_gt_8_multipass_all_m) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 9; n < 16; n++) {
    for (size_t m = 14; m <= 35; m += 7) {
      GAvgPoolMicrokernelTester().m(m).n(n).nr(8).test(
          pytorch_q8gavgpool_ukernel_mp8x7p7q__sse2);
    }
  }
}

TEST(Q8GAVGPOOL_MP8x7p7q__SSE2, n_gt_8_multipass_all_m_with_x_stride) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 9; n < 16; n++) {
    for (size_t m = 14; m <= 35; m += 7) {
      GAvgPoolMicrokernelTester().m(m).n(n).nr(8).xStride(23).test(
          pytorch_q8gavgpool_ukernel_mp8x7p7q__sse2);
    }
  }
}

TEST(Q8GAVGPOOL_UP8xM__SSE2, n_lt_8_small_m) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 1; n < 8; n++) {
    for (size_t m = 1; m < 8; m++) {
      GAvgPoolMicrokernelTester().m(m).n(n).test(
          pytorch_q8gavgpool_ukernel_up8xm__sse2);
    }
  }
}

TEST(Q8GAVGPOOL_UP8xM__SSE2, n_lt_8_large_m) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 1; n < 8; n++) {
    for (size_t m = 8; m < 16; m++) {
      GAvgPoolMicrokernelTester().m(m).n(n).test(
          pytorch_q8gavgpool_ukernel_up8xm__sse2);
    }
  }
}

TEST(Q8GAVGPOOL_UP8xM__SSE2, n_lt_8_with_x_scale) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 1; n < 8; n++) {
    for (size_t m = 1; m < 16; m += 5) {
      for (float xScale = 0.01f; xScale < 100.0f; xScale *= 3.14159265f) {
        GAvgPoolMicrokernelTester().m(m).n(n).xScale(xScale).test(
            pytorch_q8gavgpool_ukernel_up8xm__sse2);
      }
    }
  }
}

TEST(Q8GAVGPOOL_UP8xM__SSE2, n_lt_8_with_x_zero_point) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 1; n < 8; n++) {
    for (size_t m = 1; m < 16; m += 5) {
      for (int32_t xZeroPoint = 0; xZeroPoint <= 255; xZeroPoint += 51) {
        GAvgPoolMicrokernelTester()
            .m(m)
            .n(n)
            .xZeroPoint(xZeroPoint)
            .test(pytorch_q8gavgpool_ukernel_up8xm__sse2);
      }
    }
  }
}

TEST(Q8GAVGPOOL_UP8xM__SSE2, n_lt_8_with_y_scale) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 1; n < 8; n++) {
    for (size_t m = 1; m < 16; m += 5) {
      for (float yScale = 0.01f; yScale < 100.0f; yScale *= 3.14159265f) {
        GAvgPoolMicrokernelTester().m(m).n(n).yScale(yScale).test(
            pytorch_q8gavgpool_ukernel_up8xm__sse2);
      }
    }
  }
}

TEST(Q8GAVGPOOL_UP8xM__SSE2, n_lt_8_with_y_zero_point) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 1; n < 8; n++) {
    for (size_t m = 1; m < 16; m += 5) {
      for (int32_t yZeroPoint = 0; yZeroPoint <= 255; yZeroPoint += 51) {
        GAvgPoolMicrokernelTester()
            .m(m)
            .n(n)
            .yZeroPoint(yZeroPoint)
            .test(pytorch_q8gavgpool_ukernel_up8xm__sse2);
      }
    }
  }
}

TEST(Q8GAVGPOOL_UP8xM__SSE2, n_lt_8_with_y_max) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 1; n < 8; n++) {
    for (size_t m = 1; m < 16; m += 5) {
      GAvgPoolMicrokernelTester()
          .m(m)
          .n(n)
          .xZeroPoint(128)
          .yZeroPoint(128)
          .xScale(1.0f)
          .yScale(1.0f)
          .yMax(128)
          .test(pytorch_q8gavgpool_ukernel_up8xm__sse2);
    }
  }
}

TEST(Q8GAVGPOOL_UP8xM__SSE2, n_lt_8_with_y_min) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 1; n < 8; n++) {
    for (size_t m = 1; m < 16; m += 5) {
      GAvgPoolMicrokernelTester()
          .m(m)
          .n(n)
          .xZeroPoint(128)
          .yZeroPoint(128)
          .xScale(1.0f)
          .yScale(1.0f)
          .yMin(128)
          .test(pytorch_q8gavgpool_ukernel_up8xm__sse2);
    }
  }
}
#endif /* CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64 */
