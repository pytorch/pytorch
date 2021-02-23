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
#include <qnnpack/q8avgpool.h>

#include "avgpool-microkernel-tester.h"

#if CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64
TEST(Q8AVGPOOL_UP8xM__NEON, kc_lt_8_small_ks) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t kc = 1; kc < 8; kc++) {
    for (size_t ks = 1; ks < 8; ks++) {
      for (size_t kh = 1; kh <= ks; kh++) {
        for (size_t kw = 1; kw <= ks; kw++) {
          if (kh * kw == ks) {
            AvgPoolMicrokernelTester().kr(8).kh(kh).kw(kw).kc(kc).test(
                pytorch_q8avgpool_ukernel_up8xm__neon);
          }
        }
      }
    }
  }
}

TEST(Q8AVGPOOL_UP8xM__NEON, kc_lt_8_large_ks) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t kc = 1; kc < 8; kc++) {
    for (size_t ks = 8; ks < 16; ks++) {
      AvgPoolMicrokernelTester().kr(8).kh(ks).kw(1).kc(kc).test(
          pytorch_q8avgpool_ukernel_up8xm__neon);
      AvgPoolMicrokernelTester().kr(8).kh(1).kw(ks).kc(kc).test(
          pytorch_q8avgpool_ukernel_up8xm__neon);
    }
  }
}

TEST(Q8AVGPOOL_UP8xM__NEON, kc_lt_8_with_x_scale) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 1; n <= 3; n += 2) {
    for (size_t kc = 1; kc < 8; kc++) {
      for (size_t ks : std::vector<size_t>{{2, 3, 5}}) {
        for (float xScale = 0.01f; xScale < 100.0f; xScale *= 3.14159265f) {
          AvgPoolMicrokernelTester()
              .kr(8)
              .n(n)
              .kh(ks)
              .kw(ks)
              .kc(kc)
              .xScale(xScale)
              .iterations(1)
              .test(pytorch_q8avgpool_ukernel_up8xm__neon);
        }
      }
    }
  }
}

TEST(Q8AVGPOOL_UP8xM__NEON, kc_lt_8_with_x_zero_point) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 1; n <= 3; n += 2) {
    for (size_t kc = 1; kc < 8; kc++) {
      for (size_t ks : std::vector<size_t>{{2, 3, 5}}) {
        for (int32_t xZeroPoint = 0; xZeroPoint <= 255; xZeroPoint += 51) {
          AvgPoolMicrokernelTester()
              .kr(8)
              .n(n)
              .kh(ks)
              .kw(ks)
              .kc(kc)
              .xZeroPoint(uint8_t(xZeroPoint))
              .iterations(1)
              .test(pytorch_q8avgpool_ukernel_up8xm__neon);
        }
      }
    }
  }
}

TEST(Q8AVGPOOL_UP8xM__NEON, kc_lt_8_with_y_scale) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 1; n <= 3; n += 2) {
    for (size_t kc = 1; kc < 8; kc++) {
      for (size_t ks : std::vector<size_t>{{2, 3, 5}}) {
        for (float yScale = 0.01f; yScale < 100.0f; yScale *= 3.14159265f) {
          AvgPoolMicrokernelTester()
              .kr(8)
              .n(n)
              .kh(ks)
              .kw(ks)
              .kc(kc)
              .yScale(yScale)
              .iterations(1)
              .test(pytorch_q8avgpool_ukernel_up8xm__neon);
        }
      }
    }
  }
}

TEST(Q8AVGPOOL_UP8xM__NEON, kc_lt_8_with_y_zero_point) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 1; n <= 3; n += 2) {
    for (size_t kc = 1; kc < 8; kc++) {
      for (size_t ks : std::vector<size_t>{{2, 3, 5}}) {
        for (int32_t yZeroPoint = 0; yZeroPoint <= 255; yZeroPoint += 51) {
          AvgPoolMicrokernelTester()
              .kr(8)
              .n(n)
              .kh(ks)
              .kw(ks)
              .kc(kc)
              .yZeroPoint(uint8_t(yZeroPoint))
              .iterations(1)
              .test(pytorch_q8avgpool_ukernel_up8xm__neon);
        }
      }
    }
  }
}

TEST(Q8AVGPOOL_UP8xM__NEON, kc_lt_8_with_y_max) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 1; n <= 3; n += 2) {
    for (size_t kc = 1; kc < 8; kc++) {
      for (size_t ks : std::vector<size_t>{{2, 3, 5}}) {
        AvgPoolMicrokernelTester()
            .kr(8)
            .n(n)
            .kh(ks)
            .kw(ks)
            .kc(kc)
            .xZeroPoint(128)
            .yZeroPoint(128)
            .xScale(1.0f)
            .yScale(1.0f)
            .yMax(128)
            .iterations(3)
            .test(pytorch_q8avgpool_ukernel_up8xm__neon);
      }
    }
  }
}

TEST(Q8AVGPOOL_UP8xM__NEON, kc_lt_8_with_y_min) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 1; n <= 3; n += 2) {
    for (size_t kc = 1; kc < 8; kc++) {
      for (size_t ks : std::vector<size_t>{{2, 3, 5}}) {
        AvgPoolMicrokernelTester()
            .kr(8)
            .n(n)
            .kh(ks)
            .kw(ks)
            .kc(kc)
            .xZeroPoint(128)
            .yZeroPoint(128)
            .xScale(1.0f)
            .yScale(1.0f)
            .yMin(128)
            .iterations(3)
            .test(pytorch_q8avgpool_ukernel_up8xm__neon);
      }
    }
  }
}

TEST(Q8AVGPOOL_UP8xM__NEON, small_n) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 2; n < 5; n++) {
    for (size_t ks : std::vector<size_t>{{2, 3, 5}}) {
      for (size_t kc = 1; kc < 8; kc++) {
        AvgPoolMicrokernelTester()
            .kr(8)
            .n(n)
            .kh(ks)
            .kw(ks)
            .kc(kc)
            .iterations(3)
            .test(pytorch_q8avgpool_ukernel_up8xm__neon);
      }
    }
  }
}

TEST(Q8AVGPOOL_UP8xM__NEON, small_n_with_x_stride) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 2; n < 5; n++) {
    for (size_t ks : std::vector<size_t>{{2, 3, 5}}) {
      for (size_t kc = 1; kc < 8; kc++) {
        AvgPoolMicrokernelTester()
            .kr(8)
            .n(n)
            .kh(ks)
            .kw(ks)
            .kc(kc)
            .xStride(11)
            .iterations(3)
            .test(pytorch_q8avgpool_ukernel_up8xm__neon);
      }
    }
  }
}

TEST(Q8AVGPOOL_UP8xM__NEON, small_n_with_y_stride) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 2; n < 5; n++) {
    for (size_t ks : std::vector<size_t>{{2, 3, 5}}) {
      for (size_t kc = 1; kc < 8; kc++) {
        AvgPoolMicrokernelTester()
            .kr(8)
            .n(n)
            .kh(ks)
            .kw(ks)
            .kc(kc)
            .yStride(13)
            .iterations(3)
            .test(pytorch_q8avgpool_ukernel_up8xm__neon);
      }
    }
  }
}

TEST(Q8AVGPOOL_UP8xM__NEON, small_n_with_s) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 2; n < 5; n++) {
    for (size_t ks : std::vector<size_t>{{2, 3, 5}}) {
      for (size_t s = 2; s <= 5; s++) {
        for (size_t kc = 1; kc < 8; kc++) {
          AvgPoolMicrokernelTester()
              .kr(8)
              .n(n)
              .kh(ks)
              .kw(ks)
              .kc(kc)
              .s(s)
              .iterations(1)
              .test(pytorch_q8avgpool_ukernel_up8xm__neon);
        }
      }
    }
  }
}

TEST(Q8AVGPOOL_UP8x9__NEON, kc_eq_8_fulltile) {
  TEST_REQUIRES_ARM_NEON;
  auto tester = AvgPoolMicrokernelTester().kr(8).mr(9).kc(8);
  for (size_t kh = 1; kh <= tester.mr(); kh++) {
    for (size_t kw = 1; kw <= tester.mr(); kw++) {
      if (kh * kw == tester.mr()) {
        tester.kh(kh).kw(kw).test(pytorch_q8avgpool_ukernel_up8x9__neon);
      }
    }
  }
}

TEST(Q8AVGPOOL_UP8x9__NEON, kc_eq_8_subtile) {
  TEST_REQUIRES_ARM_NEON;
  auto tester = AvgPoolMicrokernelTester().kr(8).mr(9).kc(8);
  for (size_t ks = 2; ks < tester.mr(); ks++) {
    for (size_t kh = 1; kh <= ks; kh++) {
      for (size_t kw = 1; kw <= ks; kw++) {
        if (kh * kw == ks) {
          tester.kh(kh).kw(kw).test(pytorch_q8avgpool_ukernel_up8x9__neon);
        }
      }
    }
  }
}

TEST(Q8AVGPOOL_UP8x9__NEON, kc_div_8_fulltile) {
  TEST_REQUIRES_ARM_NEON;
  auto tester = AvgPoolMicrokernelTester().kr(8).mr(9);
  for (size_t kh = 1; kh <= tester.mr(); kh++) {
    for (size_t kw = 1; kw <= tester.mr(); kw++) {
      if (kh * kw == tester.mr()) {
        for (size_t kc = 8; kc < 128; kc += 24) {
          tester.kh(kh).kw(kw).kc(kc).test(pytorch_q8avgpool_ukernel_up8x9__neon);
        }
      }
    }
  }
}

TEST(Q8AVGPOOL_UP8x9__NEON, kc_div_8_subtile) {
  TEST_REQUIRES_ARM_NEON;
  auto tester = AvgPoolMicrokernelTester().kr(8).mr(9).iterations(3);
  for (size_t ks = 2; ks < tester.mr(); ks++) {
    for (size_t kh = 1; kh <= ks; kh++) {
      for (size_t kw = 1; kw <= ks; kw++) {
        if (kh * kw == ks) {
          for (size_t kc = 8; kc < 128; kc += 24) {
            tester.kh(kh).kw(kw).kc(kc).test(pytorch_q8avgpool_ukernel_up8x9__neon);
          }
        }
      }
    }
  }
}

TEST(Q8AVGPOOL_UP8x9__NEON, kc_div_8_fulltile_with_x_stride) {
  TEST_REQUIRES_ARM_NEON;
  auto tester = AvgPoolMicrokernelTester().kr(8).mr(9).iterations(3);
  for (size_t kh = 1; kh <= tester.mr(); kh++) {
    for (size_t kw = 1; kw <= tester.mr(); kw++) {
      if (kh * kw == tester.mr()) {
        for (size_t kc = 8; kc < 128; kc += 24) {
          tester.kh(kh).kw(kw).kc(kc).xStride(131).test(
              pytorch_q8avgpool_ukernel_up8x9__neon);
        }
      }
    }
  }
}

TEST(Q8AVGPOOL_UP8x9__NEON, kc_gt_8_fulltile) {
  TEST_REQUIRES_ARM_NEON;
  auto tester = AvgPoolMicrokernelTester().kr(8).mr(9);
  for (size_t kh = 1; kh <= tester.mr(); kh++) {
    for (size_t kw = 1; kw <= tester.mr(); kw++) {
      if (kh * kw == tester.mr()) {
        for (size_t kc = 9; kc < 16; kc++) {
          tester.kh(kh).kw(kw).kc(kc).test(pytorch_q8avgpool_ukernel_up8x9__neon);
        }
      }
    }
  }
}

TEST(Q8AVGPOOL_UP8x9__NEON, kc_gt_8_subtile) {
  TEST_REQUIRES_ARM_NEON;
  auto tester = AvgPoolMicrokernelTester().kr(8).mr(9).iterations(3);
  for (size_t ks = 2; ks < tester.mr(); ks++) {
    for (size_t kh = 1; kh <= ks; kh++) {
      for (size_t kw = 1; kw <= ks; kw++) {
        if (kh * kw == ks) {
          for (size_t kc = 9; kc < 16; kc++) {
            tester.kh(kh).kw(kw).kc(kc).test(pytorch_q8avgpool_ukernel_up8x9__neon);
          }
        }
      }
    }
  }
}

TEST(Q8AVGPOOL_UP8x9__NEON, kc_gt_8_fulltile_with_x_stride) {
  TEST_REQUIRES_ARM_NEON;
  auto tester = AvgPoolMicrokernelTester().kr(8).mr(9).iterations(3);
  for (size_t kh = 1; kh <= tester.mr(); kh++) {
    for (size_t kw = 1; kw <= tester.mr(); kw++) {
      if (kh * kw == tester.mr()) {
        for (size_t kc = 9; kc < 16; kc++) {
          tester.kh(kh).kw(kw).kc(kc).xStride(23).test(
              pytorch_q8avgpool_ukernel_up8x9__neon);
        }
      }
    }
  }
}

TEST(Q8AVGPOOL_UP8x9__NEON, kc_div_8_with_x_scale) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 1; n <= 5; n += 2) {
    for (size_t kc = 8; kc < 128; kc += 24) {
      for (float xScale = 0.01f; xScale < 100.0f; xScale *= 3.14159265f) {
        AvgPoolMicrokernelTester()
            .kr(8)
            .mr(9)
            .n(n)
            .kh(3)
            .kw(3)
            .kc(kc)
            .xScale(xScale)
            .iterations(2)
            .test(pytorch_q8avgpool_ukernel_up8x9__neon);
      }
    }
  }
}

TEST(Q8AVGPOOL_UP8x9__NEON, kc_div_8_with_x_zero_point) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 1; n <= 5; n += 2) {
    for (size_t kc = 8; kc < 128; kc += 24) {
      for (int32_t xZeroPoint = 0; xZeroPoint <= 255; xZeroPoint += 51) {
        AvgPoolMicrokernelTester()
            .kr(8)
            .mr(9)
            .n(n)
            .kh(3)
            .kw(3)
            .kc(kc)
            .xZeroPoint(uint8_t(xZeroPoint))
            .iterations(3)
            .test(pytorch_q8avgpool_ukernel_up8x9__neon);
      }
    }
  }
}

TEST(Q8AVGPOOL_UP8x9__NEON, kc_div_8_with_y_scale) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 1; n <= 5; n += 2) {
    for (size_t kc = 8; kc < 128; kc += 24) {
      for (float yScale = 0.01f; yScale < 100.0f; yScale *= 3.14159265f) {
        AvgPoolMicrokernelTester()
            .kr(8)
            .mr(9)
            .n(n)
            .kh(3)
            .kw(3)
            .kc(kc)
            .yScale(yScale)
            .iterations(2)
            .test(pytorch_q8avgpool_ukernel_up8x9__neon);
      }
    }
  }
}

TEST(Q8AVGPOOL_UP8x9__NEON, kc_div_8_with_y_zero_point) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 1; n <= 5; n += 2) {
    for (size_t kc = 8; kc < 128; kc += 24) {
      for (int32_t yZeroPoint = 0; yZeroPoint <= 255; yZeroPoint += 51) {
        AvgPoolMicrokernelTester()
            .kr(8)
            .mr(9)
            .n(n)
            .kh(3)
            .kw(3)
            .kc(kc)
            .yZeroPoint(uint8_t(yZeroPoint))
            .iterations(3)
            .test(pytorch_q8avgpool_ukernel_up8x9__neon);
      }
    }
  }
}

TEST(Q8AVGPOOL_UP8x9__NEON, kc_div_8_with_y_max) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 1; n <= 5; n += 2) {
    for (size_t kc = 8; kc < 128; kc += 24) {
      AvgPoolMicrokernelTester()
          .kr(8)
          .mr(9)
          .n(n)
          .kh(3)
          .kw(3)
          .kc(kc)
          .xZeroPoint(128)
          .yZeroPoint(128)
          .xScale(1.0f)
          .yScale(1.0f)
          .yMax(128)
          .test(pytorch_q8avgpool_ukernel_up8x9__neon);
    }
  }
}

TEST(Q8AVGPOOL_UP8x9__NEON, kc_div_8_with_y_min) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 1; n <= 5; n += 2) {
    for (size_t kc = 8; kc < 128; kc += 24) {
      AvgPoolMicrokernelTester()
          .kr(8)
          .mr(9)
          .n(n)
          .kh(3)
          .kw(3)
          .kc(kc)
          .xZeroPoint(128)
          .yZeroPoint(128)
          .xScale(1.0f)
          .yScale(1.0f)
          .yMin(128)
          .test(pytorch_q8avgpool_ukernel_up8x9__neon);
    }
  }
}

TEST(Q8AVGPOOL_UP8x9__NEON, small_n) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 2; n < 5; n++) {
    for (size_t ks : std::vector<size_t>{{2, 3}}) {
      for (size_t kc = 8; kc < 25; kc += 5) {
        AvgPoolMicrokernelTester().kr(8).mr(9).n(n).kh(ks).kw(ks).kc(kc).test(
            pytorch_q8avgpool_ukernel_up8x9__neon);
      }
    }
  }
}

TEST(Q8AVGPOOL_UP8x9__NEON, small_n_with_x_stride) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 2; n < 5; n++) {
    for (size_t ks : std::vector<size_t>{{2, 3}}) {
      for (size_t kc = 8; kc < 25; kc += 5) {
        AvgPoolMicrokernelTester()
            .kr(8)
            .mr(9)
            .n(n)
            .kh(ks)
            .kw(ks)
            .kc(kc)
            .xStride(29)
            .test(pytorch_q8avgpool_ukernel_up8x9__neon);
      }
    }
  }
}

TEST(Q8AVGPOOL_UP8x9__NEON, small_n_with_y_stride) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 2; n < 5; n++) {
    for (size_t ks : std::vector<size_t>{{2, 3}}) {
      for (size_t kc = 8; kc < 25; kc += 5) {
        AvgPoolMicrokernelTester()
            .kr(8)
            .mr(9)
            .n(n)
            .kh(ks)
            .kw(ks)
            .kc(kc)
            .yStride(31)
            .test(pytorch_q8avgpool_ukernel_up8x9__neon);
      }
    }
  }
}

TEST(Q8AVGPOOL_UP8x9__NEON, small_n_with_s) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 2; n < 5; n++) {
    for (size_t ks : std::vector<size_t>{{2, 3}}) {
      for (size_t kc = 8; kc < 25; kc += 5) {
        for (size_t s = 2; s <= ks; s++) {
          AvgPoolMicrokernelTester()
              .kr(8)
              .mr(9)
              .n(n)
              .kh(ks)
              .kw(ks)
              .kc(kc)
              .s(s)
              .test(pytorch_q8avgpool_ukernel_up8x9__neon);
        }
      }
    }
  }
}

TEST(Q8AVGPOOL_MP8x9P8Q__NEON, kc_eq_8_twopass_fulltile) {
  TEST_REQUIRES_ARM_NEON;
  auto tester = AvgPoolMicrokernelTester().kr(8).mr(9).qr(8).kc(8);
  const size_t ks = tester.mr() + tester.qr();
  for (size_t kh = 1; kh <= ks; kh++) {
    for (size_t kw = 1; kw <= ks; kw++) {
      if (kh * kw == ks) {
        tester.kh(kh).kw(kw).test(pytorch_q8avgpool_ukernel_mp8x9p8q__neon);
      }
    }
  }
}

TEST(Q8AVGPOOL_MP8x9P8Q__NEON, kc_eq_8_twopass_subtile) {
  TEST_REQUIRES_ARM_NEON;
  auto tester = AvgPoolMicrokernelTester().kr(8).mr(9).qr(8).kc(8);
  for (size_t ks = 10; ks < tester.mr() + tester.qr(); ks++) {
    tester.kh(ks).kw(1).test(pytorch_q8avgpool_ukernel_mp8x9p8q__neon);
    tester.kh(1).kw(ks).test(pytorch_q8avgpool_ukernel_mp8x9p8q__neon);
  }
}

TEST(Q8AVGPOOL_MP8x9P8Q__NEON, kc_eq_8_multipass_fulltile) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t ks : std::vector<size_t>{{25, 49}}) {
    auto tester = AvgPoolMicrokernelTester().kr(8).mr(9).qr(8).kc(8);
    for (size_t kh = 1; kh <= ks; kh++) {
      for (size_t kw = 1; kw <= ks; kw++) {
        if (kh * kw == ks) {
          tester.kh(kh).kw(kw).test(pytorch_q8avgpool_ukernel_mp8x9p8q__neon);
        }
      }
    }
  }
}

TEST(Q8AVGPOOL_MP8x9P8Q__NEON, kc_eq_8_multipass_subtile) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t ksMax : std::vector<size_t>{{25, 49}}) {
    auto tester = AvgPoolMicrokernelTester().kr(8).mr(9).qr(8).kc(8);
    for (size_t ks = ksMax - tester.qr() + 1; ks < ksMax; ks++) {
      tester.kh(ks).kw(1).test(pytorch_q8avgpool_ukernel_mp8x9p8q__neon);
      tester.kh(1).kw(ks).test(pytorch_q8avgpool_ukernel_mp8x9p8q__neon);
    }
  }
}

TEST(Q8AVGPOOL_MP8x9P8Q__NEON, kc_div_8_twopass_fulltile) {
  TEST_REQUIRES_ARM_NEON;
  auto tester = AvgPoolMicrokernelTester().kr(8).mr(9).qr(8).iterations(3);
  const size_t ks = 17;
  for (size_t kc = 8; kc < 128; kc += 24) {
    tester.kc(kc).kh(ks).kw(1).test(pytorch_q8avgpool_ukernel_mp8x9p8q__neon);
    tester.kc(kc).kh(1).kw(ks).test(pytorch_q8avgpool_ukernel_mp8x9p8q__neon);
  }
}

TEST(Q8AVGPOOL_MP8x9P8Q__NEON, kc_div_8_twopass_subtile) {
  TEST_REQUIRES_ARM_NEON;
  auto tester = AvgPoolMicrokernelTester().kr(8).mr(9).qr(8).iterations(3);
  for (size_t ks = 10; ks < tester.mr() + tester.qr(); ks++) {
    for (size_t kc = 8; kc < 128; kc += 24) {
      tester.kc(kc).kh(ks).kw(1).test(pytorch_q8avgpool_ukernel_mp8x9p8q__neon);
      tester.kc(kc).kh(1).kw(ks).test(pytorch_q8avgpool_ukernel_mp8x9p8q__neon);
    }
  }
}

TEST(Q8AVGPOOL_MP8x9P8Q__NEON, kc_div_8_twopass_fulltile_with_x_stride) {
  TEST_REQUIRES_ARM_NEON;
  auto tester = AvgPoolMicrokernelTester().kr(8).mr(9).qr(8).iterations(3);
  const size_t ks = tester.mr() + tester.qr();
  for (size_t kh = 1; kh <= ks; kh++) {
    for (size_t kw = 1; kw <= ks; kw++) {
      if (kh * kw == ks) {
        for (size_t kc = 8; kc < 128; kc += 24) {
          tester.kh(kh).kw(kw).kc(kc).xStride(131).test(
              pytorch_q8avgpool_ukernel_mp8x9p8q__neon);
        }
      }
    }
  }
}

TEST(Q8AVGPOOL_MP8x9P8Q__NEON, kc_div_8_multipass_fulltile) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t ks : std::vector<size_t>{{25, 49}}) {
    auto tester = AvgPoolMicrokernelTester().kr(8).mr(9).qr(8).iterations(3);
    for (size_t kh = 1; kh <= ks; kh++) {
      for (size_t kw = 1; kw <= ks; kw++) {
        if (kh * kw == ks) {
          for (size_t kc = 8; kc < 128; kc += 24) {
            tester.kh(kh).kw(kw).kc(kc).test(pytorch_q8avgpool_ukernel_mp8x9p8q__neon);
          }
        }
      }
    }
  }
}

TEST(Q8AVGPOOL_MP8x9P8Q__NEON, kc_div_8_multipass_subtile) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t ksMax : std::vector<size_t>{{25, 49}}) {
    auto tester = AvgPoolMicrokernelTester().kr(8).mr(9).qr(8).iterations(3);
    for (size_t ks = ksMax - tester.qr() + 1; ks < ksMax; ks++) {
      for (size_t kc = 8; kc < 128; kc += 24) {
        tester.kc(kc).kh(ks).kw(1).test(pytorch_q8avgpool_ukernel_mp8x9p8q__neon);
        tester.kc(kc).kh(1).kw(ks).test(pytorch_q8avgpool_ukernel_mp8x9p8q__neon);
      }
    }
  }
}

TEST(Q8AVGPOOL_MP8x9P8Q__NEON, kc_div_8_multipass_fulltile_with_x_stride) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t ks : std::vector<size_t>{{25, 49}}) {
    auto tester = AvgPoolMicrokernelTester().kr(8).mr(9).qr(8).iterations(3);
    for (size_t kh = 1; kh <= ks; kh++) {
      for (size_t kw = 1; kw <= ks; kw++) {
        if (kh * kw == ks) {
          for (size_t kc = 8; kc < 128; kc += 24) {
            tester.kh(kh).kw(kw).kc(kc).xStride(131).test(
                pytorch_q8avgpool_ukernel_mp8x9p8q__neon);
          }
        }
      }
    }
  }
}

TEST(Q8AVGPOOL_MP8x9P8Q__NEON, kc_gt_8_twopass_fulltile) {
  TEST_REQUIRES_ARM_NEON;
  auto tester = AvgPoolMicrokernelTester().kr(8).mr(9).qr(8).iterations(3);
  const size_t ks = tester.mr() + tester.qr();
  for (size_t kh = 1; kh <= ks; kh++) {
    for (size_t kw = 1; kw <= ks; kw++) {
      if (kh * kw == ks) {
        for (size_t kc = 9; kc < 16; kc++) {
          tester.kh(kh).kw(kw).kc(kc).test(pytorch_q8avgpool_ukernel_mp8x9p8q__neon);
        }
      }
    }
  }
}

TEST(Q8AVGPOOL_MP8x9P8Q__NEON, kc_gt_8_twopass_subtile) {
  TEST_REQUIRES_ARM_NEON;
  auto tester = AvgPoolMicrokernelTester().kr(8).mr(9).qr(8).iterations(3);
  for (size_t ks = 10; ks < tester.mr() + tester.qr(); ks++) {
    for (size_t kc = 9; kc < 16; kc++) {
      tester.kc(kc).kh(ks).kw(1).test(pytorch_q8avgpool_ukernel_mp8x9p8q__neon);
      tester.kc(kc).kh(1).kw(ks).test(pytorch_q8avgpool_ukernel_mp8x9p8q__neon);
    }
  }
}

TEST(Q8AVGPOOL_MP8x9P8Q__NEON, kc_gt_8_twopass_fulltile_with_x_stride) {
  TEST_REQUIRES_ARM_NEON;
  auto tester = AvgPoolMicrokernelTester().kr(8).mr(9).qr(8).iterations(3);
  const size_t ks = tester.mr() + tester.qr();
  for (size_t kh = 1; kh <= ks; kh++) {
    for (size_t kw = 1; kw <= ks; kw++) {
      if (kh * kw == ks) {
        for (size_t kc = 9; kc < 16; kc++) {
          tester.kh(kh).kw(kw).kc(kc).xStride(23).test(
              pytorch_q8avgpool_ukernel_mp8x9p8q__neon);
        }
      }
    }
  }
}

TEST(Q8AVGPOOL_MP8x9P8Q__NEON, kc_gt_8_multipass_fulltile) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t ks : std::vector<size_t>{{25, 49}}) {
    auto tester = AvgPoolMicrokernelTester().kr(8).mr(9).qr(8).iterations(3);
    for (size_t kh = 1; kh <= ks; kh++) {
      for (size_t kw = 1; kw <= ks; kw++) {
        if (kh * kw == ks) {
          for (size_t kc = 9; kc < 16; kc++) {
            tester.kh(kh).kw(kw).kc(kc).test(pytorch_q8avgpool_ukernel_mp8x9p8q__neon);
          }
        }
      }
    }
  }
}

TEST(Q8AVGPOOL_MP8x9P8Q__NEON, kc_gt_8_multipass_subtile) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t ksMax : std::vector<size_t>{{25, 49}}) {
    auto tester = AvgPoolMicrokernelTester().kr(8).mr(9).qr(8).iterations(3);
    for (size_t ks = ksMax - tester.qr() + 1; ks < ksMax; ks++) {
      for (size_t kc = 9; kc < 16; kc++) {
        tester.kc(kc).kh(ks).kw(1).test(pytorch_q8avgpool_ukernel_mp8x9p8q__neon);
        tester.kc(kc).kh(1).kw(ks).test(pytorch_q8avgpool_ukernel_mp8x9p8q__neon);
      }
    }
  }
}

TEST(Q8AVGPOOL_MP8x9P8Q__NEON, kc_gt_8_multipass_fulltile_with_x_stride) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t ks : std::vector<size_t>{{25, 49}}) {
    auto tester = AvgPoolMicrokernelTester().kr(8).mr(9).qr(8).iterations(3);
    for (size_t kh = 1; kh <= ks; kh++) {
      for (size_t kw = 1; kw <= ks; kw++) {
        if (kh * kw == ks) {
          for (size_t kc = 9; kc < 16; kc++) {
            tester.kh(kh).kw(kw).kc(kc).xStride(23).test(
                pytorch_q8avgpool_ukernel_mp8x9p8q__neon);
          }
        }
      }
    }
  }
}

TEST(Q8AVGPOOL_MP8x9P8Q__NEON, kc_div_8_with_x_scale) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 1; n <= 5; n += 2) {
    for (size_t kc = 8; kc < 128; kc += 24) {
      for (float xScale = 0.01f; xScale < 100.0f; xScale *= 3.14159265f) {
        AvgPoolMicrokernelTester()
            .kr(8)
            .mr(9)
            .qr(8)
            .n(n)
            .kh(5)
            .kw(5)
            .kc(kc)
            .xScale(xScale)
            .iterations(1)
            .test(pytorch_q8avgpool_ukernel_mp8x9p8q__neon);
      }
    }
  }
}

TEST(Q8AVGPOOL_MP8x9P8Q__NEON, kc_div_8_with_x_zero_point) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 1; n <= 5; n += 2) {
    for (size_t kc = 8; kc < 128; kc += 24) {
      for (int32_t xZeroPoint = 0; xZeroPoint <= 255; xZeroPoint += 51) {
        AvgPoolMicrokernelTester()
            .kr(8)
            .mr(9)
            .qr(8)
            .n(n)
            .kh(5)
            .kw(5)
            .kc(kc)
            .xZeroPoint(uint8_t(xZeroPoint))
            .iterations(1)
            .test(pytorch_q8avgpool_ukernel_mp8x9p8q__neon);
      }
    }
  }
}

TEST(Q8AVGPOOL_MP8x9P8Q__NEON, kc_div_8_with_y_scale) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 1; n <= 5; n += 2) {
    for (size_t kc = 8; kc < 128; kc += 24) {
      for (float yScale = 0.01f; yScale < 100.0f; yScale *= 3.14159265f) {
        AvgPoolMicrokernelTester()
            .kr(8)
            .mr(9)
            .qr(8)
            .n(n)
            .kh(5)
            .kw(5)
            .kc(kc)
            .yScale(yScale)
            .iterations(1)
            .test(pytorch_q8avgpool_ukernel_mp8x9p8q__neon);
      }
    }
  }
}

TEST(Q8AVGPOOL_MP8x9P8Q__NEON, kc_div_8_with_y_zero_point) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 1; n <= 5; n += 2) {
    for (size_t kc = 8; kc < 128; kc += 24) {
      for (int32_t yZeroPoint = 0; yZeroPoint <= 255; yZeroPoint += 51) {
        AvgPoolMicrokernelTester()
            .kr(8)
            .mr(9)
            .qr(8)
            .n(n)
            .kh(5)
            .kw(5)
            .kc(kc)
            .yZeroPoint(uint8_t(yZeroPoint))
            .iterations(1)
            .test(pytorch_q8avgpool_ukernel_mp8x9p8q__neon);
      }
    }
  }
}

TEST(Q8AVGPOOL_MP8x9P8Q__NEON, kc_div_8_with_y_max) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 1; n <= 5; n += 2) {
    for (size_t kc = 8; kc < 128; kc += 24) {
      AvgPoolMicrokernelTester()
          .kr(8)
          .mr(9)
          .qr(8)
          .n(n)
          .kh(5)
          .kw(5)
          .kc(kc)
          .xZeroPoint(128)
          .yZeroPoint(128)
          .xScale(1.0f)
          .yScale(1.0f)
          .yMax(128)
          .iterations(3)
          .test(pytorch_q8avgpool_ukernel_mp8x9p8q__neon);
    }
  }
}

TEST(Q8AVGPOOL_MP8x9P8Q__NEON, kc_div_8_with_y_min) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 1; n <= 5; n += 2) {
    for (size_t kc = 8; kc < 128; kc += 24) {
      AvgPoolMicrokernelTester()
          .kr(8)
          .mr(9)
          .qr(8)
          .n(n)
          .kh(5)
          .kw(5)
          .kc(kc)
          .xZeroPoint(128)
          .yZeroPoint(128)
          .xScale(1.0f)
          .yScale(1.0f)
          .yMin(128)
          .iterations(3)
          .test(pytorch_q8avgpool_ukernel_mp8x9p8q__neon);
    }
  }
}

TEST(Q8AVGPOOL_MP8x9P8Q__NEON, small_n) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 2; n < 5; n++) {
    for (size_t ks : std::vector<size_t>{{5, 7}}) {
      for (size_t kc = 8; kc < 25; kc += 5) {
        AvgPoolMicrokernelTester()
            .kr(8)
            .mr(9)
            .qr(8)
            .n(n)
            .kh(ks)
            .kw(ks)
            .kc(kc)
            .test(pytorch_q8avgpool_ukernel_mp8x9p8q__neon);
      }
    }
  }
}

TEST(Q8AVGPOOL_MP8x9P8Q__NEON, small_n_with_x_stride) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 2; n < 5; n++) {
    for (size_t ks : std::vector<size_t>{{5, 7}}) {
      for (size_t kc = 8; kc < 25; kc += 5) {
        AvgPoolMicrokernelTester()
            .kr(8)
            .mr(9)
            .qr(8)
            .n(n)
            .kh(ks)
            .kw(ks)
            .kc(kc)
            .xStride(29)
            .test(pytorch_q8avgpool_ukernel_mp8x9p8q__neon);
      }
    }
  }
}

TEST(Q8AVGPOOL_MP8x9P8Q__NEON, small_n_with_y_stride) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 2; n < 5; n++) {
    for (size_t ks : std::vector<size_t>{{5, 7}}) {
      for (size_t kc = 8; kc < 25; kc += 5) {
        AvgPoolMicrokernelTester()
            .kr(8)
            .mr(9)
            .qr(8)
            .n(n)
            .kh(ks)
            .kw(ks)
            .kc(kc)
            .yStride(31)
            .test(pytorch_q8avgpool_ukernel_mp8x9p8q__neon);
      }
    }
  }
}

TEST(Q8AVGPOOL_MP8x9P8Q__NEON, small_n_with_s) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 2; n < 5; n++) {
    for (size_t ks : std::vector<size_t>{{5, 7}}) {
      for (size_t s = 2; s <= 5; s++) {
        for (size_t kc = 8; kc < 25; kc += 5) {
          AvgPoolMicrokernelTester()
              .kr(8)
              .mr(9)
              .qr(8)
              .n(n)
              .kh(ks)
              .kw(ks)
              .kc(kc)
              .s(s)
              .test(pytorch_q8avgpool_ukernel_mp8x9p8q__neon);
        }
      }
    }
  }
}
#endif /* CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64 */

#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
TEST(Q8AVGPOOL_UP8xM__SSE2, kc_lt_8_small_ks) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t kc = 1; kc < 8; kc++) {
    for (size_t ks = 1; ks < 8; ks++) {
      for (size_t kh = 1; kh <= ks; kh++) {
        for (size_t kw = 1; kw <= ks; kw++) {
          if (kh * kw == ks) {
            AvgPoolMicrokernelTester().kr(8).kh(kh).kw(kw).kc(kc).test(
                pytorch_q8avgpool_ukernel_up8xm__sse2);
          }
        }
      }
    }
  }
}

TEST(Q8AVGPOOL_UP8xM__SSE2, kc_lt_8_large_ks) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t kc = 1; kc < 8; kc++) {
    for (size_t ks = 8; ks < 16; ks++) {
      AvgPoolMicrokernelTester().kr(8).kh(ks).kw(1).kc(kc).test(
          pytorch_q8avgpool_ukernel_up8xm__sse2);
      AvgPoolMicrokernelTester().kr(8).kh(1).kw(ks).kc(kc).test(
          pytorch_q8avgpool_ukernel_up8xm__sse2);
    }
  }
}

TEST(Q8AVGPOOL_UP8xM__SSE2, kc_lt_8_with_x_scale) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 1; n <= 3; n += 2) {
    for (size_t kc = 1; kc < 8; kc++) {
      for (size_t ks : std::vector<size_t>{{2, 3, 5}}) {
        for (float xScale = 0.01f; xScale < 100.0f; xScale *= 3.14159265f) {
          AvgPoolMicrokernelTester()
              .kr(8)
              .n(n)
              .kh(ks)
              .kw(ks)
              .kc(kc)
              .xScale(xScale)
              .iterations(1)
              .test(pytorch_q8avgpool_ukernel_up8xm__sse2);
        }
      }
    }
  }
}

TEST(Q8AVGPOOL_UP8xM__SSE2, kc_lt_8_with_x_zero_point) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 1; n <= 3; n += 2) {
    for (size_t kc = 1; kc < 8; kc++) {
      for (size_t ks : std::vector<size_t>{{2, 3, 5}}) {
        for (int32_t xZeroPoint = 0; xZeroPoint <= 255; xZeroPoint += 51) {
          AvgPoolMicrokernelTester()
              .kr(8)
              .n(n)
              .kh(ks)
              .kw(ks)
              .kc(kc)
              .xZeroPoint(uint8_t(xZeroPoint))
              .iterations(1)
              .test(pytorch_q8avgpool_ukernel_up8xm__sse2);
        }
      }
    }
  }
}

TEST(Q8AVGPOOL_UP8xM__SSE2, kc_lt_8_with_y_scale) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 1; n <= 3; n += 2) {
    for (size_t kc = 1; kc < 8; kc++) {
      for (size_t ks : std::vector<size_t>{{2, 3, 5}}) {
        for (float yScale = 0.01f; yScale < 100.0f; yScale *= 3.14159265f) {
          AvgPoolMicrokernelTester()
              .kr(8)
              .n(n)
              .kh(ks)
              .kw(ks)
              .kc(kc)
              .yScale(yScale)
              .iterations(1)
              .test(pytorch_q8avgpool_ukernel_up8xm__sse2);
        }
      }
    }
  }
}

TEST(Q8AVGPOOL_UP8xM__SSE2, kc_lt_8_with_y_zero_point) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 1; n <= 3; n += 2) {
    for (size_t kc = 1; kc < 8; kc++) {
      for (size_t ks : std::vector<size_t>{{2, 3, 5}}) {
        for (int32_t yZeroPoint = 0; yZeroPoint <= 255; yZeroPoint += 51) {
          AvgPoolMicrokernelTester()
              .kr(8)
              .n(n)
              .kh(ks)
              .kw(ks)
              .kc(kc)
              .yZeroPoint(uint8_t(yZeroPoint))
              .iterations(1)
              .test(pytorch_q8avgpool_ukernel_up8xm__sse2);
        }
      }
    }
  }
}

TEST(Q8AVGPOOL_UP8xM__SSE2, kc_lt_8_with_y_max) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 1; n <= 3; n += 2) {
    for (size_t kc = 1; kc < 8; kc++) {
      for (size_t ks : std::vector<size_t>{{2, 3, 5}}) {
        AvgPoolMicrokernelTester()
            .kr(8)
            .n(n)
            .kh(ks)
            .kw(ks)
            .kc(kc)
            .xZeroPoint(128)
            .yZeroPoint(128)
            .xScale(1.0f)
            .yScale(1.0f)
            .yMax(128)
            .iterations(3)
            .test(pytorch_q8avgpool_ukernel_up8xm__sse2);
      }
    }
  }
}

TEST(Q8AVGPOOL_UP8xM__SSE2, kc_lt_8_with_y_min) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 1; n <= 3; n += 2) {
    for (size_t kc = 1; kc < 8; kc++) {
      for (size_t ks : std::vector<size_t>{{2, 3, 5}}) {
        AvgPoolMicrokernelTester()
            .kr(8)
            .n(n)
            .kh(ks)
            .kw(ks)
            .kc(kc)
            .xZeroPoint(128)
            .yZeroPoint(128)
            .xScale(1.0f)
            .yScale(1.0f)
            .yMin(128)
            .iterations(3)
            .test(pytorch_q8avgpool_ukernel_up8xm__sse2);
      }
    }
  }
}

TEST(Q8AVGPOOL_UP8xM__SSE2, small_n) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 2; n < 5; n++) {
    for (size_t ks : std::vector<size_t>{{2, 3, 5}}) {
      for (size_t kc = 1; kc < 8; kc++) {
        AvgPoolMicrokernelTester()
            .kr(8)
            .n(n)
            .kh(ks)
            .kw(ks)
            .kc(kc)
            .iterations(3)
            .test(pytorch_q8avgpool_ukernel_up8xm__sse2);
      }
    }
  }
}

TEST(Q8AVGPOOL_UP8xM__SSE2, small_n_with_x_stride) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 2; n < 5; n++) {
    for (size_t ks : std::vector<size_t>{{2, 3, 5}}) {
      for (size_t kc = 1; kc < 8; kc++) {
        AvgPoolMicrokernelTester()
            .kr(8)
            .n(n)
            .kh(ks)
            .kw(ks)
            .kc(kc)
            .xStride(11)
            .iterations(3)
            .test(pytorch_q8avgpool_ukernel_up8xm__sse2);
      }
    }
  }
}

TEST(Q8AVGPOOL_UP8xM__SSE2, small_n_with_y_stride) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 2; n < 5; n++) {
    for (size_t ks : std::vector<size_t>{{2, 3, 5}}) {
      for (size_t kc = 1; kc < 8; kc++) {
        AvgPoolMicrokernelTester()
            .kr(8)
            .n(n)
            .kh(ks)
            .kw(ks)
            .kc(kc)
            .yStride(13)
            .iterations(3)
            .test(pytorch_q8avgpool_ukernel_up8xm__sse2);
      }
    }
  }
}

TEST(Q8AVGPOOL_UP8xM__SSE2, small_n_with_s) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 2; n < 5; n++) {
    for (size_t ks : std::vector<size_t>{{2, 3, 5}}) {
      for (size_t s = 2; s <= 5; s++) {
        for (size_t kc = 1; kc < 8; kc++) {
          AvgPoolMicrokernelTester()
              .kr(8)
              .n(n)
              .kh(ks)
              .kw(ks)
              .kc(kc)
              .s(s)
              .iterations(1)
              .test(pytorch_q8avgpool_ukernel_up8xm__sse2);
        }
      }
    }
  }
}

TEST(Q8AVGPOOL_UP8x9__SSE2, kc_eq_8_fulltile) {
  TEST_REQUIRES_X86_SSE2;
  auto tester = AvgPoolMicrokernelTester().kr(8).mr(9).kc(8);
  for (size_t kh = 1; kh <= tester.mr(); kh++) {
    for (size_t kw = 1; kw <= tester.mr(); kw++) {
      if (kh * kw == tester.mr()) {
        tester.kh(kh).kw(kw).test(pytorch_q8avgpool_ukernel_up8x9__sse2);
      }
    }
  }
}

TEST(Q8AVGPOOL_UP8x9__SSE2, kc_eq_8_subtile) {
  TEST_REQUIRES_X86_SSE2;
  auto tester = AvgPoolMicrokernelTester().kr(8).mr(9).kc(8);
  for (size_t ks = 2; ks < tester.mr(); ks++) {
    for (size_t kh = 1; kh <= ks; kh++) {
      for (size_t kw = 1; kw <= ks; kw++) {
        if (kh * kw == ks) {
          tester.kh(kh).kw(kw).test(pytorch_q8avgpool_ukernel_up8x9__sse2);
        }
      }
    }
  }
}

TEST(Q8AVGPOOL_UP8x9__SSE2, kc_div_8_fulltile) {
  TEST_REQUIRES_X86_SSE2;
  auto tester = AvgPoolMicrokernelTester().kr(8).mr(9);
  for (size_t kh = 1; kh <= tester.mr(); kh++) {
    for (size_t kw = 1; kw <= tester.mr(); kw++) {
      if (kh * kw == tester.mr()) {
        for (size_t kc = 8; kc < 128; kc += 24) {
          tester.kh(kh).kw(kw).kc(kc).test(pytorch_q8avgpool_ukernel_up8x9__sse2);
        }
      }
    }
  }
}

TEST(Q8AVGPOOL_UP8x9__SSE2, kc_div_8_subtile) {
  TEST_REQUIRES_X86_SSE2;
  auto tester = AvgPoolMicrokernelTester().kr(8).mr(9).iterations(3);
  for (size_t ks = 2; ks < tester.mr(); ks++) {
    for (size_t kh = 1; kh <= ks; kh++) {
      for (size_t kw = 1; kw <= ks; kw++) {
        if (kh * kw == ks) {
          for (size_t kc = 8; kc < 128; kc += 24) {
            tester.kh(kh).kw(kw).kc(kc).test(pytorch_q8avgpool_ukernel_up8x9__sse2);
          }
        }
      }
    }
  }
}

TEST(Q8AVGPOOL_UP8x9__SSE2, kc_div_8_fulltile_with_x_stride) {
  TEST_REQUIRES_X86_SSE2;
  auto tester = AvgPoolMicrokernelTester().kr(8).mr(9).iterations(3);
  for (size_t kh = 1; kh <= tester.mr(); kh++) {
    for (size_t kw = 1; kw <= tester.mr(); kw++) {
      if (kh * kw == tester.mr()) {
        for (size_t kc = 8; kc < 128; kc += 24) {
          tester.kh(kh).kw(kw).kc(kc).xStride(131).test(
              pytorch_q8avgpool_ukernel_up8x9__sse2);
        }
      }
    }
  }
}

TEST(Q8AVGPOOL_UP8x9__SSE2, kc_gt_8_fulltile) {
  TEST_REQUIRES_X86_SSE2;
  auto tester = AvgPoolMicrokernelTester().kr(8).mr(9);
  for (size_t kh = 1; kh <= tester.mr(); kh++) {
    for (size_t kw = 1; kw <= tester.mr(); kw++) {
      if (kh * kw == tester.mr()) {
        for (size_t kc = 9; kc < 16; kc++) {
          tester.kh(kh).kw(kw).kc(kc).test(pytorch_q8avgpool_ukernel_up8x9__sse2);
        }
      }
    }
  }
}

TEST(Q8AVGPOOL_UP8x9__SSE2, kc_gt_8_subtile) {
  TEST_REQUIRES_X86_SSE2;
  auto tester = AvgPoolMicrokernelTester().kr(8).mr(9).iterations(3);
  for (size_t ks = 2; ks < tester.mr(); ks++) {
    for (size_t kh = 1; kh <= ks; kh++) {
      for (size_t kw = 1; kw <= ks; kw++) {
        if (kh * kw == ks) {
          for (size_t kc = 9; kc < 16; kc++) {
            tester.kh(kh).kw(kw).kc(kc).test(pytorch_q8avgpool_ukernel_up8x9__sse2);
          }
        }
      }
    }
  }
}

TEST(Q8AVGPOOL_UP8x9__SSE2, kc_gt_8_fulltile_with_x_stride) {
  TEST_REQUIRES_X86_SSE2;
  auto tester = AvgPoolMicrokernelTester().kr(8).mr(9).iterations(3);
  for (size_t kh = 1; kh <= tester.mr(); kh++) {
    for (size_t kw = 1; kw <= tester.mr(); kw++) {
      if (kh * kw == tester.mr()) {
        for (size_t kc = 9; kc < 16; kc++) {
          tester.kh(kh).kw(kw).kc(kc).xStride(23).test(
              pytorch_q8avgpool_ukernel_up8x9__sse2);
        }
      }
    }
  }
}

TEST(Q8AVGPOOL_UP8x9__SSE2, kc_div_8_with_x_scale) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 1; n <= 5; n += 2) {
    for (size_t kc = 8; kc < 128; kc += 24) {
      for (float xScale = 0.01f; xScale < 100.0f; xScale *= 3.14159265f) {
        AvgPoolMicrokernelTester()
            .kr(8)
            .mr(9)
            .n(n)
            .kh(3)
            .kw(3)
            .kc(kc)
            .xScale(xScale)
            .iterations(2)
            .test(pytorch_q8avgpool_ukernel_up8x9__sse2);
      }
    }
  }
}

TEST(Q8AVGPOOL_UP8x9__SSE2, kc_div_8_with_x_zero_point) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 1; n <= 5; n += 2) {
    for (size_t kc = 8; kc < 128; kc += 24) {
      for (int32_t xZeroPoint = 0; xZeroPoint <= 255; xZeroPoint += 51) {
        AvgPoolMicrokernelTester()
            .kr(8)
            .mr(9)
            .n(n)
            .kh(3)
            .kw(3)
            .kc(kc)
            .xZeroPoint(uint8_t(xZeroPoint))
            .iterations(3)
            .test(pytorch_q8avgpool_ukernel_up8x9__sse2);
      }
    }
  }
}

TEST(Q8AVGPOOL_UP8x9__SSE2, kc_div_8_with_y_scale) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 1; n <= 5; n += 2) {
    for (size_t kc = 8; kc < 128; kc += 24) {
      for (float yScale = 0.01f; yScale < 100.0f; yScale *= 3.14159265f) {
        AvgPoolMicrokernelTester()
            .kr(8)
            .mr(9)
            .n(n)
            .kh(3)
            .kw(3)
            .kc(kc)
            .yScale(yScale)
            .iterations(2)
            .test(pytorch_q8avgpool_ukernel_up8x9__sse2);
      }
    }
  }
}

TEST(Q8AVGPOOL_UP8x9__SSE2, kc_div_8_with_y_zero_point) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 1; n <= 5; n += 2) {
    for (size_t kc = 8; kc < 128; kc += 24) {
      for (int32_t yZeroPoint = 0; yZeroPoint <= 255; yZeroPoint += 51) {
        AvgPoolMicrokernelTester()
            .kr(8)
            .mr(9)
            .n(n)
            .kh(3)
            .kw(3)
            .kc(kc)
            .yZeroPoint(uint8_t(yZeroPoint))
            .iterations(3)
            .test(pytorch_q8avgpool_ukernel_up8x9__sse2);
      }
    }
  }
}

TEST(Q8AVGPOOL_UP8x9__SSE2, kc_div_8_with_y_max) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 1; n <= 5; n += 2) {
    for (size_t kc = 8; kc < 128; kc += 24) {
      AvgPoolMicrokernelTester()
          .kr(8)
          .mr(9)
          .n(n)
          .kh(3)
          .kw(3)
          .kc(kc)
          .xZeroPoint(128)
          .yZeroPoint(128)
          .xScale(1.0f)
          .yScale(1.0f)
          .yMax(128)
          .test(pytorch_q8avgpool_ukernel_up8x9__sse2);
    }
  }
}

TEST(Q8AVGPOOL_UP8x9__SSE2, kc_div_8_with_y_min) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 1; n <= 5; n += 2) {
    for (size_t kc = 8; kc < 128; kc += 24) {
      AvgPoolMicrokernelTester()
          .kr(8)
          .mr(9)
          .n(n)
          .kh(3)
          .kw(3)
          .kc(kc)
          .xZeroPoint(128)
          .yZeroPoint(128)
          .xScale(1.0f)
          .yScale(1.0f)
          .yMin(128)
          .test(pytorch_q8avgpool_ukernel_up8x9__sse2);
    }
  }
}

TEST(Q8AVGPOOL_UP8x9__SSE2, small_n) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 2; n < 5; n++) {
    for (size_t ks : std::vector<size_t>{{2, 3}}) {
      for (size_t kc = 8; kc < 25; kc += 5) {
        AvgPoolMicrokernelTester().kr(8).mr(9).n(n).kh(ks).kw(ks).kc(kc).test(
            pytorch_q8avgpool_ukernel_up8x9__sse2);
      }
    }
  }
}

TEST(Q8AVGPOOL_UP8x9__SSE2, small_n_with_x_stride) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 2; n < 5; n++) {
    for (size_t ks : std::vector<size_t>{{2, 3}}) {
      for (size_t kc = 8; kc < 25; kc += 5) {
        AvgPoolMicrokernelTester()
            .kr(8)
            .mr(9)
            .n(n)
            .kh(ks)
            .kw(ks)
            .kc(kc)
            .xStride(29)
            .test(pytorch_q8avgpool_ukernel_up8x9__sse2);
      }
    }
  }
}

TEST(Q8AVGPOOL_UP8x9__SSE2, small_n_with_y_stride) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 2; n < 5; n++) {
    for (size_t ks : std::vector<size_t>{{2, 3}}) {
      for (size_t kc = 8; kc < 25; kc += 5) {
        AvgPoolMicrokernelTester()
            .kr(8)
            .mr(9)
            .n(n)
            .kh(ks)
            .kw(ks)
            .kc(kc)
            .yStride(31)
            .test(pytorch_q8avgpool_ukernel_up8x9__sse2);
      }
    }
  }
}

TEST(Q8AVGPOOL_UP8x9__SSE2, small_n_with_s) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 2; n < 5; n++) {
    for (size_t ks : std::vector<size_t>{{2, 3}}) {
      for (size_t kc = 8; kc < 25; kc += 5) {
        for (size_t s = 2; s <= ks; s++) {
          AvgPoolMicrokernelTester()
              .kr(8)
              .mr(9)
              .n(n)
              .kh(ks)
              .kw(ks)
              .kc(kc)
              .s(s)
              .test(pytorch_q8avgpool_ukernel_up8x9__sse2);
        }
      }
    }
  }
}

TEST(Q8AVGPOOL_MP8x9P8Q__SSE2, kc_eq_8_twopass_fulltile) {
  TEST_REQUIRES_X86_SSE2;
  auto tester = AvgPoolMicrokernelTester().kr(8).mr(9).qr(8).kc(8);
  const size_t ks = tester.mr() + tester.qr();
  for (size_t kh = 1; kh <= ks; kh++) {
    for (size_t kw = 1; kw <= ks; kw++) {
      if (kh * kw == ks) {
        tester.kh(kh).kw(kw).test(pytorch_q8avgpool_ukernel_mp8x9p8q__sse2);
      }
    }
  }
}

TEST(Q8AVGPOOL_MP8x9P8Q__SSE2, kc_eq_8_twopass_subtile) {
  TEST_REQUIRES_X86_SSE2;
  auto tester = AvgPoolMicrokernelTester().kr(8).mr(9).qr(8).kc(8);
  for (size_t ks = 10; ks < tester.mr() + tester.qr(); ks++) {
    tester.kh(ks).kw(1).test(pytorch_q8avgpool_ukernel_mp8x9p8q__sse2);
    tester.kh(1).kw(ks).test(pytorch_q8avgpool_ukernel_mp8x9p8q__sse2);
  }
}

TEST(Q8AVGPOOL_MP8x9P8Q__SSE2, kc_eq_8_multipass_fulltile) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t ks : std::vector<size_t>{{25, 49}}) {
    auto tester = AvgPoolMicrokernelTester().kr(8).mr(9).qr(8).kc(8);
    for (size_t kh = 1; kh <= ks; kh++) {
      for (size_t kw = 1; kw <= ks; kw++) {
        if (kh * kw == ks) {
          tester.kh(kh).kw(kw).test(pytorch_q8avgpool_ukernel_mp8x9p8q__sse2);
        }
      }
    }
  }
}

TEST(Q8AVGPOOL_MP8x9P8Q__SSE2, kc_eq_8_multipass_subtile) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t ksMax : std::vector<size_t>{{25, 49}}) {
    auto tester = AvgPoolMicrokernelTester().kr(8).mr(9).qr(8).kc(8);
    for (size_t ks = ksMax - tester.qr() + 1; ks < ksMax; ks++) {
      tester.kh(ks).kw(1).test(pytorch_q8avgpool_ukernel_mp8x9p8q__sse2);
      tester.kh(1).kw(ks).test(pytorch_q8avgpool_ukernel_mp8x9p8q__sse2);
    }
  }
}

TEST(Q8AVGPOOL_MP8x9P8Q__SSE2, kc_div_8_twopass_fulltile) {
  TEST_REQUIRES_X86_SSE2;
  auto tester = AvgPoolMicrokernelTester().kr(8).mr(9).qr(8).iterations(3);
  const size_t ks = 17;
  for (size_t kc = 8; kc < 128; kc += 24) {
    tester.kc(kc).kh(ks).kw(1).test(pytorch_q8avgpool_ukernel_mp8x9p8q__sse2);
    tester.kc(kc).kh(1).kw(ks).test(pytorch_q8avgpool_ukernel_mp8x9p8q__sse2);
  }
}

TEST(Q8AVGPOOL_MP8x9P8Q__SSE2, kc_div_8_twopass_subtile) {
  TEST_REQUIRES_X86_SSE2;
  auto tester = AvgPoolMicrokernelTester().kr(8).mr(9).qr(8).iterations(3);
  for (size_t ks = 10; ks < tester.mr() + tester.qr(); ks++) {
    for (size_t kc = 8; kc < 128; kc += 24) {
      tester.kc(kc).kh(ks).kw(1).test(pytorch_q8avgpool_ukernel_mp8x9p8q__sse2);
      tester.kc(kc).kh(1).kw(ks).test(pytorch_q8avgpool_ukernel_mp8x9p8q__sse2);
    }
  }
}

TEST(Q8AVGPOOL_MP8x9P8Q__SSE2, kc_div_8_twopass_fulltile_with_x_stride) {
  TEST_REQUIRES_X86_SSE2;
  auto tester = AvgPoolMicrokernelTester().kr(8).mr(9).qr(8).iterations(3);
  const size_t ks = tester.mr() + tester.qr();
  for (size_t kh = 1; kh <= ks; kh++) {
    for (size_t kw = 1; kw <= ks; kw++) {
      if (kh * kw == ks) {
        for (size_t kc = 8; kc < 128; kc += 24) {
          tester.kh(kh).kw(kw).kc(kc).xStride(131).test(
              pytorch_q8avgpool_ukernel_mp8x9p8q__sse2);
        }
      }
    }
  }
}

TEST(Q8AVGPOOL_MP8x9P8Q__SSE2, kc_div_8_multipass_fulltile) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t ks : std::vector<size_t>{{25, 49}}) {
    auto tester = AvgPoolMicrokernelTester().kr(8).mr(9).qr(8).iterations(3);
    for (size_t kh = 1; kh <= ks; kh++) {
      for (size_t kw = 1; kw <= ks; kw++) {
        if (kh * kw == ks) {
          for (size_t kc = 8; kc < 128; kc += 24) {
            tester.kh(kh).kw(kw).kc(kc).test(pytorch_q8avgpool_ukernel_mp8x9p8q__sse2);
          }
        }
      }
    }
  }
}

TEST(Q8AVGPOOL_MP8x9P8Q__SSE2, kc_div_8_multipass_subtile) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t ksMax : std::vector<size_t>{{25, 49}}) {
    auto tester = AvgPoolMicrokernelTester().kr(8).mr(9).qr(8).iterations(3);
    for (size_t ks = ksMax - tester.qr() + 1; ks < ksMax; ks++) {
      for (size_t kc = 8; kc < 128; kc += 24) {
        tester.kc(kc).kh(ks).kw(1).test(pytorch_q8avgpool_ukernel_mp8x9p8q__sse2);
        tester.kc(kc).kh(1).kw(ks).test(pytorch_q8avgpool_ukernel_mp8x9p8q__sse2);
      }
    }
  }
}

TEST(Q8AVGPOOL_MP8x9P8Q__SSE2, kc_div_8_multipass_fulltile_with_x_stride) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t ks : std::vector<size_t>{{25, 49}}) {
    auto tester = AvgPoolMicrokernelTester().kr(8).mr(9).qr(8).iterations(3);
    for (size_t kh = 1; kh <= ks; kh++) {
      for (size_t kw = 1; kw <= ks; kw++) {
        if (kh * kw == ks) {
          for (size_t kc = 8; kc < 128; kc += 24) {
            tester.kh(kh).kw(kw).kc(kc).xStride(131).test(
                pytorch_q8avgpool_ukernel_mp8x9p8q__sse2);
          }
        }
      }
    }
  }
}

TEST(Q8AVGPOOL_MP8x9P8Q__SSE2, kc_gt_8_twopass_fulltile) {
  TEST_REQUIRES_X86_SSE2;
  auto tester = AvgPoolMicrokernelTester().kr(8).mr(9).qr(8).iterations(3);
  const size_t ks = tester.mr() + tester.qr();
  for (size_t kh = 1; kh <= ks; kh++) {
    for (size_t kw = 1; kw <= ks; kw++) {
      if (kh * kw == ks) {
        for (size_t kc = 9; kc < 16; kc++) {
          tester.kh(kh).kw(kw).kc(kc).test(pytorch_q8avgpool_ukernel_mp8x9p8q__sse2);
        }
      }
    }
  }
}

TEST(Q8AVGPOOL_MP8x9P8Q__SSE2, kc_gt_8_twopass_subtile) {
  TEST_REQUIRES_X86_SSE2;
  auto tester = AvgPoolMicrokernelTester().kr(8).mr(9).qr(8).iterations(3);
  for (size_t ks = 10; ks < tester.mr() + tester.qr(); ks++) {
    for (size_t kc = 9; kc < 16; kc++) {
      tester.kc(kc).kh(ks).kw(1).test(pytorch_q8avgpool_ukernel_mp8x9p8q__sse2);
      tester.kc(kc).kh(1).kw(ks).test(pytorch_q8avgpool_ukernel_mp8x9p8q__sse2);
    }
  }
}

TEST(Q8AVGPOOL_MP8x9P8Q__SSE2, kc_gt_8_twopass_fulltile_with_x_stride) {
  TEST_REQUIRES_X86_SSE2;
  auto tester = AvgPoolMicrokernelTester().kr(8).mr(9).qr(8).iterations(3);
  const size_t ks = tester.mr() + tester.qr();
  for (size_t kh = 1; kh <= ks; kh++) {
    for (size_t kw = 1; kw <= ks; kw++) {
      if (kh * kw == ks) {
        for (size_t kc = 9; kc < 16; kc++) {
          tester.kh(kh).kw(kw).kc(kc).xStride(23).test(
              pytorch_q8avgpool_ukernel_mp8x9p8q__sse2);
        }
      }
    }
  }
}

TEST(Q8AVGPOOL_MP8x9P8Q__SSE2, kc_gt_8_multipass_fulltile) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t ks : std::vector<size_t>{{25, 49}}) {
    auto tester = AvgPoolMicrokernelTester().kr(8).mr(9).qr(8).iterations(3);
    for (size_t kh = 1; kh <= ks; kh++) {
      for (size_t kw = 1; kw <= ks; kw++) {
        if (kh * kw == ks) {
          for (size_t kc = 9; kc < 16; kc++) {
            tester.kh(kh).kw(kw).kc(kc).test(pytorch_q8avgpool_ukernel_mp8x9p8q__sse2);
          }
        }
      }
    }
  }
}

TEST(Q8AVGPOOL_MP8x9P8Q__SSE2, kc_gt_8_multipass_subtile) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t ksMax : std::vector<size_t>{{25, 49}}) {
    auto tester = AvgPoolMicrokernelTester().kr(8).mr(9).qr(8).iterations(3);
    for (size_t ks = ksMax - tester.qr() + 1; ks < ksMax; ks++) {
      for (size_t kc = 9; kc < 16; kc++) {
        tester.kc(kc).kh(ks).kw(1).test(pytorch_q8avgpool_ukernel_mp8x9p8q__sse2);
        tester.kc(kc).kh(1).kw(ks).test(pytorch_q8avgpool_ukernel_mp8x9p8q__sse2);
      }
    }
  }
}

TEST(Q8AVGPOOL_MP8x9P8Q__SSE2, kc_gt_8_multipass_fulltile_with_x_stride) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t ks : std::vector<size_t>{{25, 49}}) {
    auto tester = AvgPoolMicrokernelTester().kr(8).mr(9).qr(8).iterations(3);
    for (size_t kh = 1; kh <= ks; kh++) {
      for (size_t kw = 1; kw <= ks; kw++) {
        if (kh * kw == ks) {
          for (size_t kc = 9; kc < 16; kc++) {
            tester.kh(kh).kw(kw).kc(kc).xStride(23).test(
                pytorch_q8avgpool_ukernel_mp8x9p8q__sse2);
          }
        }
      }
    }
  }
}

TEST(Q8AVGPOOL_MP8x9P8Q__SSE2, kc_div_8_with_x_scale) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 1; n <= 5; n += 2) {
    for (size_t kc = 8; kc < 128; kc += 24) {
      for (float xScale = 0.01f; xScale < 100.0f; xScale *= 3.14159265f) {
        AvgPoolMicrokernelTester()
            .kr(8)
            .mr(9)
            .qr(8)
            .n(n)
            .kh(5)
            .kw(5)
            .kc(kc)
            .xScale(xScale)
            .iterations(1)
            .test(pytorch_q8avgpool_ukernel_mp8x9p8q__sse2);
      }
    }
  }
}

TEST(Q8AVGPOOL_MP8x9P8Q__SSE2, kc_div_8_with_x_zero_point) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 1; n <= 5; n += 2) {
    for (size_t kc = 8; kc < 128; kc += 24) {
      for (int32_t xZeroPoint = 0; xZeroPoint <= 255; xZeroPoint += 51) {
        AvgPoolMicrokernelTester()
            .kr(8)
            .mr(9)
            .qr(8)
            .n(n)
            .kh(5)
            .kw(5)
            .kc(kc)
            .xZeroPoint(uint8_t(xZeroPoint))
            .iterations(1)
            .test(pytorch_q8avgpool_ukernel_mp8x9p8q__sse2);
      }
    }
  }
}

TEST(Q8AVGPOOL_MP8x9P8Q__SSE2, kc_div_8_with_y_scale) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 1; n <= 5; n += 2) {
    for (size_t kc = 8; kc < 128; kc += 24) {
      for (float yScale = 0.01f; yScale < 100.0f; yScale *= 3.14159265f) {
        AvgPoolMicrokernelTester()
            .kr(8)
            .mr(9)
            .qr(8)
            .n(n)
            .kh(5)
            .kw(5)
            .kc(kc)
            .yScale(yScale)
            .iterations(1)
            .test(pytorch_q8avgpool_ukernel_mp8x9p8q__sse2);
      }
    }
  }
}

TEST(Q8AVGPOOL_MP8x9P8Q__SSE2, kc_div_8_with_y_zero_point) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 1; n <= 5; n += 2) {
    for (size_t kc = 8; kc < 128; kc += 24) {
      for (int32_t yZeroPoint = 0; yZeroPoint <= 255; yZeroPoint += 51) {
        AvgPoolMicrokernelTester()
            .kr(8)
            .mr(9)
            .qr(8)
            .n(n)
            .kh(5)
            .kw(5)
            .kc(kc)
            .yZeroPoint(uint8_t(yZeroPoint))
            .iterations(1)
            .test(pytorch_q8avgpool_ukernel_mp8x9p8q__sse2);
      }
    }
  }
}

TEST(Q8AVGPOOL_MP8x9P8Q__SSE2, kc_div_8_with_y_max) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 1; n <= 5; n += 2) {
    for (size_t kc = 8; kc < 128; kc += 24) {
      AvgPoolMicrokernelTester()
          .kr(8)
          .mr(9)
          .qr(8)
          .n(n)
          .kh(5)
          .kw(5)
          .kc(kc)
          .xZeroPoint(128)
          .yZeroPoint(128)
          .xScale(1.0f)
          .yScale(1.0f)
          .yMax(128)
          .iterations(3)
          .test(pytorch_q8avgpool_ukernel_mp8x9p8q__sse2);
    }
  }
}

TEST(Q8AVGPOOL_MP8x9P8Q__SSE2, kc_div_8_with_y_min) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 1; n <= 5; n += 2) {
    for (size_t kc = 8; kc < 128; kc += 24) {
      AvgPoolMicrokernelTester()
          .kr(8)
          .mr(9)
          .qr(8)
          .n(n)
          .kh(5)
          .kw(5)
          .kc(kc)
          .xZeroPoint(128)
          .yZeroPoint(128)
          .xScale(1.0f)
          .yScale(1.0f)
          .yMin(128)
          .iterations(3)
          .test(pytorch_q8avgpool_ukernel_mp8x9p8q__sse2);
    }
  }
}

TEST(Q8AVGPOOL_MP8x9P8Q__SSE2, small_n) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 2; n < 5; n++) {
    for (size_t ks : std::vector<size_t>{{5, 7}}) {
      for (size_t kc = 8; kc < 25; kc += 5) {
        AvgPoolMicrokernelTester()
            .kr(8)
            .mr(9)
            .qr(8)
            .n(n)
            .kh(ks)
            .kw(ks)
            .kc(kc)
            .test(pytorch_q8avgpool_ukernel_mp8x9p8q__sse2);
      }
    }
  }
}

TEST(Q8AVGPOOL_MP8x9P8Q__SSE2, small_n_with_x_stride) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 2; n < 5; n++) {
    for (size_t ks : std::vector<size_t>{{5, 7}}) {
      for (size_t kc = 8; kc < 25; kc += 5) {
        AvgPoolMicrokernelTester()
            .kr(8)
            .mr(9)
            .qr(8)
            .n(n)
            .kh(ks)
            .kw(ks)
            .kc(kc)
            .xStride(29)
            .test(pytorch_q8avgpool_ukernel_mp8x9p8q__sse2);
      }
    }
  }
}

TEST(Q8AVGPOOL_MP8x9P8Q__SSE2, small_n_with_y_stride) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 2; n < 5; n++) {
    for (size_t ks : std::vector<size_t>{{5, 7}}) {
      for (size_t kc = 8; kc < 25; kc += 5) {
        AvgPoolMicrokernelTester()
            .kr(8)
            .mr(9)
            .qr(8)
            .n(n)
            .kh(ks)
            .kw(ks)
            .kc(kc)
            .yStride(31)
            .test(pytorch_q8avgpool_ukernel_mp8x9p8q__sse2);
      }
    }
  }
}

TEST(Q8AVGPOOL_MP8x9P8Q__SSE2, small_n_with_s) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 2; n < 5; n++) {
    for (size_t ks : std::vector<size_t>{{5, 7}}) {
      for (size_t s = 2; s <= 5; s++) {
        for (size_t kc = 8; kc < 25; kc += 5) {
          AvgPoolMicrokernelTester()
              .kr(8)
              .mr(9)
              .qr(8)
              .n(n)
              .kh(ks)
              .kw(ks)
              .kc(kc)
              .s(s)
              .test(pytorch_q8avgpool_ukernel_mp8x9p8q__sse2);
        }
      }
    }
  }
}
#endif /* CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64 */
