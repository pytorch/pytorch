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
#include <qnnpack/u8maxpool.h>

#include "maxpool-microkernel-tester.h"

#if CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64
TEST(U8MAXPOOL_SUB16__NEON, kc_lt_16_mx1_pool) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t kc = 1; kc < 16; kc++) {
    for (size_t ks = 2; ks < 16; ks++) {
      MaxPoolMicrokernelTester().kr(16).kh(ks).kw(1).kc(kc).test(
          pytorch_u8maxpool_ukernel_sub16__neon);
    }
  }
}

TEST(U8MAXPOOL_SUB16__NEON, kc_lt_16_mx1_pool_with_qmin) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t kc = 1; kc < 16; kc++) {
    for (size_t ks = 2; ks < 16; ks++) {
      MaxPoolMicrokernelTester().kr(16).kh(ks).kw(1).kc(kc).qmin(192).test(
          pytorch_u8maxpool_ukernel_sub16__neon);
    }
  }
}

TEST(U8MAXPOOL_SUB16__NEON, kc_lt_16_mx1_pool_with_qmax) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t kc = 1; kc < 16; kc++) {
    for (size_t ks = 2; ks < 16; ks++) {
      MaxPoolMicrokernelTester().kr(16).kh(ks).kw(1).kc(kc).qmax(192).test(
          pytorch_u8maxpool_ukernel_sub16__neon);
    }
  }
}

TEST(U8MAXPOOL_SUB16__NEON, kc_lt_16_1xm_pool) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t kc = 1; kc < 16; kc++) {
    for (size_t ks = 2; ks < 16; ks++) {
      MaxPoolMicrokernelTester().kr(16).kh(1).kw(ks).kc(kc).test(
          pytorch_u8maxpool_ukernel_sub16__neon);
    }
  }
}

TEST(U8MAXPOOL_SUB16__NEON, kc_lt_16_1xm_pool_with_qmin) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t kc = 1; kc < 16; kc++) {
    for (size_t ks = 2; ks < 16; ks++) {
      MaxPoolMicrokernelTester().kr(16).kh(1).kw(ks).kc(kc).qmin(192).test(
          pytorch_u8maxpool_ukernel_sub16__neon);
    }
  }
}

TEST(U8MAXPOOL_SUB16__NEON, kc_lt_16_1xm_pool_with_qmax) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t kc = 1; kc < 16; kc++) {
    for (size_t ks = 2; ks < 16; ks++) {
      MaxPoolMicrokernelTester().kr(16).kh(1).kw(ks).kc(kc).qmax(192).test(
          pytorch_u8maxpool_ukernel_sub16__neon);
    }
  }
}

TEST(U8MAXPOOL_SUB16__NEON, kc_lt_16_small_n) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 2; n < 5; n++) {
    for (size_t ks : std::vector<size_t>{{2, 3, 5}}) {
      for (size_t kc = 1; kc < 16; kc++) {
        MaxPoolMicrokernelTester()
            .kr(16)
            .n(n)
            .kh(ks)
            .kw(ks)
            .kc(kc)
            .iterations(3)
            .test(pytorch_u8maxpool_ukernel_sub16__neon);
      }
    }
  }
}

TEST(U8MAXPOOL_SUB16__NEON, kc_lt_16_small_n_with_x_stride) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 2; n < 5; n++) {
    for (size_t ks : std::vector<size_t>{{2, 3, 5}}) {
      for (size_t kc = 1; kc < 16; kc++) {
        MaxPoolMicrokernelTester()
            .kr(16)
            .n(n)
            .kh(ks)
            .kw(ks)
            .kc(kc)
            .xStride(17)
            .iterations(3)
            .test(pytorch_u8maxpool_ukernel_sub16__neon);
      }
    }
  }
}

TEST(U8MAXPOOL_SUB16__NEON, kc_lt_16_small_n_with_s) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 2; n < 5; n++) {
    for (size_t ks : std::vector<size_t>{{2, 3, 5}}) {
      for (size_t s = 2; s <= 5; s++) {
        for (size_t kc = 1; kc < 16; kc++) {
          MaxPoolMicrokernelTester()
              .kr(16)
              .n(n)
              .kh(ks)
              .kw(ks)
              .kc(kc)
              .s(s)
              .iterations(1)
              .test(pytorch_u8maxpool_ukernel_sub16__neon);
        }
      }
    }
  }
}

TEST(U8MAXPOOL_SUB16__NEON, kc_lt_16_small_n_with_qmin) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 2; n < 5; n++) {
    for (size_t ks : std::vector<size_t>{{2, 3, 5}}) {
      for (size_t kc = 1; kc < 16; kc++) {
        MaxPoolMicrokernelTester()
            .kr(16)
            .n(n)
            .kh(ks)
            .kw(ks)
            .kc(kc)
            .qmin(192)
            .iterations(3)
            .test(pytorch_u8maxpool_ukernel_sub16__neon);
      }
    }
  }
}

TEST(U8MAXPOOL_SUB16__NEON, kc_lt_16_small_n_with_qmax) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 2; n < 5; n++) {
    for (size_t ks : std::vector<size_t>{{2, 3, 5}}) {
      for (size_t kc = 1; kc < 16; kc++) {
        MaxPoolMicrokernelTester()
            .kr(16)
            .n(n)
            .kh(ks)
            .kw(ks)
            .kc(kc)
            .qmax(192)
            .iterations(3)
            .test(pytorch_u8maxpool_ukernel_sub16__neon);
      }
    }
  }
}

TEST(U8MAXPOOL_16x9P8Q__NEON, kc_eq_16_unipass_fulltile) {
  TEST_REQUIRES_ARM_NEON;
  auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).kc(16);
  for (size_t kh = 1; kh <= tester.mr(); kh++) {
    for (size_t kw = 1; kw <= tester.mr(); kw++) {
      if (kh * kw == tester.mr()) {
        tester.kh(kh).kw(kw).test(pytorch_u8maxpool_ukernel_16x9p8q__neon);
      }
    }
  }
}

TEST(U8MAXPOOL_16x9P8Q__NEON, kc_eq_16_unipass_fulltile_with_qmin) {
  TEST_REQUIRES_ARM_NEON;
  auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).kc(16);
  for (size_t kh = 1; kh <= tester.mr(); kh++) {
    for (size_t kw = 1; kw <= tester.mr(); kw++) {
      if (kh * kw == tester.mr()) {
        tester.kh(kh).kw(kw).qmin(192).test(pytorch_u8maxpool_ukernel_16x9p8q__neon);
      }
    }
  }
}

TEST(U8MAXPOOL_16x9P8Q__NEON, kc_eq_16_unipass_fulltile_with_qmax) {
  TEST_REQUIRES_ARM_NEON;
  auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).kc(16);
  for (size_t kh = 1; kh <= tester.mr(); kh++) {
    for (size_t kw = 1; kw <= tester.mr(); kw++) {
      if (kh * kw == tester.mr()) {
        tester.kh(kh).kw(kw).qmax(192).test(pytorch_u8maxpool_ukernel_16x9p8q__neon);
      }
    }
  }
}

TEST(U8MAXPOOL_16x9P8Q__NEON, kc_eq_16_unipass_subtile) {
  TEST_REQUIRES_ARM_NEON;
  auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).kc(16);
  for (size_t ks = 2; ks < tester.mr(); ks++) {
    tester.kh(ks).kw(1).test(pytorch_u8maxpool_ukernel_16x9p8q__neon);
    tester.kh(1).kw(ks).test(pytorch_u8maxpool_ukernel_16x9p8q__neon);
  }
}

TEST(U8MAXPOOL_16x9P8Q__NEON, kc_div_16_unipass_fulltile) {
  TEST_REQUIRES_ARM_NEON;
  auto tester = MaxPoolMicrokernelTester().kr(16).mr(9);
  for (size_t kh = 1; kh <= tester.mr(); kh++) {
    for (size_t kw = 1; kw <= tester.mr(); kw++) {
      if (kh * kw == tester.mr()) {
        for (size_t kc = 16; kc < 256; kc += 48) {
          tester.kh(kh).kw(kw).kc(kc).test(pytorch_u8maxpool_ukernel_16x9p8q__neon);
        }
      }
    }
  }
}

TEST(U8MAXPOOL_16x9P8Q__NEON, kc_div_16_unipass_fulltile_with_qmin) {
  TEST_REQUIRES_ARM_NEON;
  auto tester = MaxPoolMicrokernelTester().kr(16).mr(9);
  for (size_t kh = 1; kh <= tester.mr(); kh++) {
    for (size_t kw = 1; kw <= tester.mr(); kw++) {
      if (kh * kw == tester.mr()) {
        for (size_t kc = 16; kc < 256; kc += 48) {
          tester.kh(kh).kw(kw).kc(kc).qmin(192).test(
              pytorch_u8maxpool_ukernel_16x9p8q__neon);
        }
      }
    }
  }
}

TEST(U8MAXPOOL_16x9P8Q__NEON, kc_div_16_unipass_fulltile_with_qmax) {
  TEST_REQUIRES_ARM_NEON;
  auto tester = MaxPoolMicrokernelTester().kr(16).mr(9);
  for (size_t kh = 1; kh <= tester.mr(); kh++) {
    for (size_t kw = 1; kw <= tester.mr(); kw++) {
      if (kh * kw == tester.mr()) {
        for (size_t kc = 16; kc < 256; kc += 48) {
          tester.kh(kh).kw(kw).kc(kc).qmax(192).test(
              pytorch_u8maxpool_ukernel_16x9p8q__neon);
        }
      }
    }
  }
}

TEST(U8MAXPOOL_16x9P8Q__NEON, kc_div_16_unipass_fulltile_with_x_stride) {
  TEST_REQUIRES_ARM_NEON;
  auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).iterations(3);
  for (size_t kh = 1; kh <= tester.mr(); kh++) {
    for (size_t kw = 1; kw <= tester.mr(); kw++) {
      if (kh * kw == tester.mr()) {
        for (size_t kc = 16; kc < 256; kc += 48) {
          tester.kh(kh).kw(kw).kc(kc).xStride(257).test(
              pytorch_u8maxpool_ukernel_16x9p8q__neon);
        }
      }
    }
  }
}

TEST(U8MAXPOOL_16x9P8Q__NEON, kc_div_16_unipass_subtile) {
  TEST_REQUIRES_ARM_NEON;
  auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).iterations(3);
  for (size_t ks = 2; ks < tester.mr(); ks++) {
    for (size_t kc = 16; kc < 256; kc += 48) {
      tester.kh(ks).kw(1).kc(kc).test(pytorch_u8maxpool_ukernel_16x9p8q__neon);
      tester.kh(1).kw(ks).kc(kc).test(pytorch_u8maxpool_ukernel_16x9p8q__neon);
    }
  }
}

TEST(U8MAXPOOL_16x9P8Q__NEON, kc_gt_16_unipass_fulltile) {
  TEST_REQUIRES_ARM_NEON;
  auto tester = MaxPoolMicrokernelTester().kr(16).mr(9);
  for (size_t kh = 1; kh <= tester.mr(); kh++) {
    for (size_t kw = 1; kw <= tester.mr(); kw++) {
      if (kh * kw == tester.mr()) {
        for (size_t kc = 17; kc < 32; kc++) {
          tester.kh(kh).kw(kw).kc(kc).test(pytorch_u8maxpool_ukernel_16x9p8q__neon);
        }
      }
    }
  }
}

TEST(U8MAXPOOL_16x9P8Q__NEON, kc_gt_16_unipass_fulltile_with_qmin) {
  TEST_REQUIRES_ARM_NEON;
  auto tester = MaxPoolMicrokernelTester().kr(16).mr(9);
  for (size_t kh = 1; kh <= tester.mr(); kh++) {
    for (size_t kw = 1; kw <= tester.mr(); kw++) {
      if (kh * kw == tester.mr()) {
        for (size_t kc = 17; kc < 32; kc++) {
          tester.kh(kh).kw(kw).kc(kc).qmin(192).test(
              pytorch_u8maxpool_ukernel_16x9p8q__neon);
        }
      }
    }
  }
}

TEST(U8MAXPOOL_16x9P8Q__NEON, kc_gt_16_unipass_fulltile_with_qmax) {
  TEST_REQUIRES_ARM_NEON;
  auto tester = MaxPoolMicrokernelTester().kr(16).mr(9);
  for (size_t kh = 1; kh <= tester.mr(); kh++) {
    for (size_t kw = 1; kw <= tester.mr(); kw++) {
      if (kh * kw == tester.mr()) {
        for (size_t kc = 17; kc < 32; kc++) {
          tester.kh(kh).kw(kw).kc(kc).qmax(192).test(
              pytorch_u8maxpool_ukernel_16x9p8q__neon);
        }
      }
    }
  }
}

TEST(U8MAXPOOL_16x9P8Q__NEON, kc_gt_16_unipass_fulltile_with_x_stride) {
  TEST_REQUIRES_ARM_NEON;
  auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).iterations(3);
  for (size_t kh = 1; kh <= tester.mr(); kh++) {
    for (size_t kw = 1; kw <= tester.mr(); kw++) {
      if (kh * kw == tester.mr()) {
        for (size_t kc = 17; kc < 32; kc++) {
          tester.kh(kh).kw(kw).kc(kc).xStride(257).test(
              pytorch_u8maxpool_ukernel_16x9p8q__neon);
        }
      }
    }
  }
}

TEST(U8MAXPOOL_16x9P8Q__NEON, kc_gt_16_unipass_subtile) {
  TEST_REQUIRES_ARM_NEON;
  auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).iterations(3);
  for (size_t ks = 2; ks < tester.mr(); ks++) {
    for (size_t kc = 17; kc < 32; kc++) {
      tester.kh(ks).kw(1).kc(kc).test(pytorch_u8maxpool_ukernel_16x9p8q__neon);
      tester.kh(1).kw(ks).kc(kc).test(pytorch_u8maxpool_ukernel_16x9p8q__neon);
    }
  }
}

TEST(U8MAXPOOL_16x9P8Q__NEON, kc_eq_16_twopass_fulltile) {
  TEST_REQUIRES_ARM_NEON;
  auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8).kc(16);
  for (size_t kh = 1; kh <= tester.mr() + tester.qr(); kh++) {
    for (size_t kw = 1; kw <= tester.mr() + tester.qr(); kw++) {
      if (kh * kw == tester.mr() + tester.qr()) {
        tester.kh(kh).kw(kw).test(pytorch_u8maxpool_ukernel_16x9p8q__neon);
      }
    }
  }
}

TEST(U8MAXPOOL_16x9P8Q__NEON, kc_eq_16_twopass_fulltile_with_qmin) {
  TEST_REQUIRES_ARM_NEON;
  auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8).kc(16);
  for (size_t kh = 1; kh <= tester.mr() + tester.qr(); kh++) {
    for (size_t kw = 1; kw <= tester.mr() + tester.qr(); kw++) {
      if (kh * kw == tester.mr() + tester.qr()) {
        tester.kh(kh).kw(kw).qmin(192).test(pytorch_u8maxpool_ukernel_16x9p8q__neon);
      }
    }
  }
}

TEST(U8MAXPOOL_16x9P8Q__NEON, kc_eq_16_twopass_fulltile_with_qmax) {
  TEST_REQUIRES_ARM_NEON;
  auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8).kc(16);
  for (size_t kh = 1; kh <= tester.mr() + tester.qr(); kh++) {
    for (size_t kw = 1; kw <= tester.mr() + tester.qr(); kw++) {
      if (kh * kw == tester.mr() + tester.qr()) {
        tester.kh(kh).kw(kw).qmax(192).test(pytorch_u8maxpool_ukernel_16x9p8q__neon);
      }
    }
  }
}

TEST(U8MAXPOOL_16x9P8Q__NEON, kc_eq_16_twopass_subtile) {
  TEST_REQUIRES_ARM_NEON;
  auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8).kc(16);
  for (size_t ks = tester.mr() + 1; ks < tester.mr() + tester.qr(); ks++) {
    tester.kh(ks).kw(1).test(pytorch_u8maxpool_ukernel_16x9p8q__neon);
    tester.kh(1).kw(ks).test(pytorch_u8maxpool_ukernel_16x9p8q__neon);
  }
}

TEST(U8MAXPOOL_16x9P8Q__NEON, kc_div_16_twopass_fulltile) {
  TEST_REQUIRES_ARM_NEON;
  auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8);
  for (size_t kh = 1; kh <= tester.mr() + tester.qr(); kh++) {
    for (size_t kw = 1; kw <= tester.mr() + tester.qr(); kw++) {
      if (kh * kw == tester.mr() + tester.qr()) {
        for (size_t kc = 16; kc < 256; kc += 48) {
          tester.kh(kh).kw(kw).kc(kc).test(pytorch_u8maxpool_ukernel_16x9p8q__neon);
        }
      }
    }
  }
}

TEST(U8MAXPOOL_16x9P8Q__NEON, kc_div_16_twopass_fulltile_with_qmin) {
  TEST_REQUIRES_ARM_NEON;
  auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8);
  for (size_t kh = 1; kh <= tester.mr() + tester.qr(); kh++) {
    for (size_t kw = 1; kw <= tester.mr() + tester.qr(); kw++) {
      if (kh * kw == tester.mr() + tester.qr()) {
        for (size_t kc = 16; kc < 256; kc += 48) {
          tester.kh(kh).kw(kw).kc(kc).qmin(192).test(
              pytorch_u8maxpool_ukernel_16x9p8q__neon);
        }
      }
    }
  }
}

TEST(U8MAXPOOL_16x9P8Q__NEON, kc_div_16_twopass_fulltile_with_qmax) {
  TEST_REQUIRES_ARM_NEON;
  auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8);
  for (size_t kh = 1; kh <= tester.mr() + tester.qr(); kh++) {
    for (size_t kw = 1; kw <= tester.mr() + tester.qr(); kw++) {
      if (kh * kw == tester.mr() + tester.qr()) {
        for (size_t kc = 16; kc < 256; kc += 48) {
          tester.kh(kh).kw(kw).kc(kc).qmax(192).test(
              pytorch_u8maxpool_ukernel_16x9p8q__neon);
        }
      }
    }
  }
}

TEST(U8MAXPOOL_16x9P8Q__NEON, kc_div_16_twopass_fulltile_with_x_stride) {
  TEST_REQUIRES_ARM_NEON;
  auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8).iterations(3);
  for (size_t kh = 1; kh <= tester.mr() + tester.qr(); kh++) {
    for (size_t kw = 1; kw <= tester.mr() + tester.qr(); kw++) {
      if (kh * kw == tester.mr() + tester.qr()) {
        for (size_t kc = 16; kc < 256; kc += 48) {
          tester.kh(kh).kw(kw).kc(kc).xStride(257).test(
              pytorch_u8maxpool_ukernel_16x9p8q__neon);
        }
      }
    }
  }
}

TEST(U8MAXPOOL_16x9P8Q__NEON, kc_div_16_twopass_subtile) {
  TEST_REQUIRES_ARM_NEON;
  auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8).iterations(3);
  for (size_t ks = tester.mr() + 1; ks < tester.mr() + tester.qr(); ks++) {
    for (size_t kc = 16; kc < 256; kc += 48) {
      tester.kh(ks).kw(1).kc(kc).test(pytorch_u8maxpool_ukernel_16x9p8q__neon);
      tester.kh(1).kw(ks).kc(kc).test(pytorch_u8maxpool_ukernel_16x9p8q__neon);
    }
  }
}

TEST(U8MAXPOOL_16x9P8Q__NEON, kc_gt_16_twopass_fulltile) {
  TEST_REQUIRES_ARM_NEON;
  auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8);
  for (size_t kh = 1; kh <= tester.mr() + tester.qr(); kh++) {
    for (size_t kw = 1; kw <= tester.mr() + tester.qr(); kw++) {
      if (kh * kw == tester.mr() + tester.qr()) {
        for (size_t kc = 17; kc < 32; kc++) {
          tester.kh(kh).kw(kw).kc(kc).test(pytorch_u8maxpool_ukernel_16x9p8q__neon);
        }
      }
    }
  }
}

TEST(U8MAXPOOL_16x9P8Q__NEON, kc_gt_16_twopass_fulltile_with_qmin) {
  TEST_REQUIRES_ARM_NEON;
  auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8);
  for (size_t kh = 1; kh <= tester.mr() + tester.qr(); kh++) {
    for (size_t kw = 1; kw <= tester.mr() + tester.qr(); kw++) {
      if (kh * kw == tester.mr() + tester.qr()) {
        for (size_t kc = 17; kc < 32; kc++) {
          tester.kh(kh).kw(kw).kc(kc).qmin(192).test(
              pytorch_u8maxpool_ukernel_16x9p8q__neon);
        }
      }
    }
  }
}

TEST(U8MAXPOOL_16x9P8Q__NEON, kc_gt_16_twopass_fulltile_with_qmax) {
  TEST_REQUIRES_ARM_NEON;
  auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8);
  for (size_t kh = 1; kh <= tester.mr() + tester.qr(); kh++) {
    for (size_t kw = 1; kw <= tester.mr() + tester.qr(); kw++) {
      if (kh * kw == tester.mr() + tester.qr()) {
        for (size_t kc = 17; kc < 32; kc++) {
          tester.kh(kh).kw(kw).kc(kc).qmax(192).test(
              pytorch_u8maxpool_ukernel_16x9p8q__neon);
        }
      }
    }
  }
}

TEST(U8MAXPOOL_16x9P8Q__NEON, kc_gt_16_twopass_fulltile_with_x_stride) {
  TEST_REQUIRES_ARM_NEON;
  auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8).iterations(3);
  for (size_t kh = 1; kh <= tester.mr() + tester.qr(); kh++) {
    for (size_t kw = 1; kw <= tester.mr() + tester.qr(); kw++) {
      if (kh * kw == tester.mr() + tester.qr()) {
        for (size_t kc = 17; kc < 32; kc++) {
          tester.kh(kh).kw(kw).kc(kc).xStride(257).test(
              pytorch_u8maxpool_ukernel_16x9p8q__neon);
        }
      }
    }
  }
}

TEST(U8MAXPOOL_16x9P8Q__NEON, kc_gt_16_twopass_subtile) {
  TEST_REQUIRES_ARM_NEON;
  auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8).iterations(3);
  for (size_t ks = tester.mr() + 1; ks < tester.mr() + tester.qr(); ks++) {
    for (size_t kc = 17; kc < 32; kc++) {
      tester.kh(ks).kw(1).kc(kc).test(pytorch_u8maxpool_ukernel_16x9p8q__neon);
      tester.kh(1).kw(ks).kc(kc).test(pytorch_u8maxpool_ukernel_16x9p8q__neon);
    }
  }
}

TEST(U8MAXPOOL_16x9P8Q__NEON, kc_eq_16_multipass) {
  TEST_REQUIRES_ARM_NEON;
  auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8).kc(16);
  for (size_t ks = tester.mr() + tester.qr() + 1;
       ks < tester.mr() + 3 * tester.qr();
       ks += 3) {
    tester.kh(ks).kw(1).test(pytorch_u8maxpool_ukernel_16x9p8q__neon);
    tester.kh(1).kw(ks).test(pytorch_u8maxpool_ukernel_16x9p8q__neon);
  }
}

TEST(U8MAXPOOL_16x9P8Q__NEON, kc_eq_16_multipass_with_qmin) {
  TEST_REQUIRES_ARM_NEON;
  auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8).kc(16);
  for (size_t ks = tester.mr() + tester.qr() + 1;
       ks < tester.mr() + 3 * tester.qr();
       ks += 3) {
    tester.kh(ks).kw(1).qmin(192).test(pytorch_u8maxpool_ukernel_16x9p8q__neon);
    tester.kh(1).kw(ks).qmin(192).test(pytorch_u8maxpool_ukernel_16x9p8q__neon);
  }
}

TEST(U8MAXPOOL_16x9P8Q__NEON, kc_eq_16_multipass_with_qmax) {
  TEST_REQUIRES_ARM_NEON;
  auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8).kc(16);
  for (size_t ks = tester.mr() + tester.qr() + 1;
       ks < tester.mr() + 3 * tester.qr();
       ks += 3) {
    tester.kh(ks).kw(1).qmax(192).test(pytorch_u8maxpool_ukernel_16x9p8q__neon);
    tester.kh(1).kw(ks).qmax(192).test(pytorch_u8maxpool_ukernel_16x9p8q__neon);
  }
}

TEST(U8MAXPOOL_16x9P8Q__NEON, kc_div_16_multipass) {
  TEST_REQUIRES_ARM_NEON;
  auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8);
  for (size_t ks = tester.mr() + tester.qr() + 1;
       ks < tester.mr() + 3 * tester.qr();
       ks += 3) {
    for (size_t kc = 16; kc < 256; kc += 48) {
      tester.kh(ks).kw(1).kc(kc).test(pytorch_u8maxpool_ukernel_16x9p8q__neon);
      tester.kh(1).kw(ks).kc(kc).test(pytorch_u8maxpool_ukernel_16x9p8q__neon);
    }
  }
}

TEST(U8MAXPOOL_16x9P8Q__NEON, kc_div_16_multipass_with_qmin) {
  TEST_REQUIRES_ARM_NEON;
  auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8);
  for (size_t ks = tester.mr() + tester.qr() + 1;
       ks < tester.mr() + 3 * tester.qr();
       ks += 3) {
    for (size_t kc = 16; kc < 256; kc += 48) {
      tester.kh(ks).kw(1).kc(kc).qmin(192).test(
          pytorch_u8maxpool_ukernel_16x9p8q__neon);
      tester.kh(1).kw(ks).kc(kc).qmin(192).test(
          pytorch_u8maxpool_ukernel_16x9p8q__neon);
    }
  }
}

TEST(U8MAXPOOL_16x9P8Q__NEON, kc_div_16_multipass_with_qmax) {
  TEST_REQUIRES_ARM_NEON;
  auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8);
  for (size_t ks = tester.mr() + tester.qr() + 1;
       ks < tester.mr() + 3 * tester.qr();
       ks += 3) {
    for (size_t kc = 16; kc < 256; kc += 48) {
      tester.kh(ks).kw(1).kc(kc).qmax(192).test(
          pytorch_u8maxpool_ukernel_16x9p8q__neon);
      tester.kh(1).kw(ks).kc(kc).qmax(192).test(
          pytorch_u8maxpool_ukernel_16x9p8q__neon);
    }
  }
}

TEST(U8MAXPOOL_16x9P8Q__NEON, kc_div_16_multipass_with_x_stride) {
  TEST_REQUIRES_ARM_NEON;
  auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8).iterations(3);
  for (size_t ks = tester.mr() + tester.qr() + 1;
       ks < tester.mr() + 3 * tester.qr();
       ks += 3) {
    for (size_t kc = 16; kc < 256; kc += 48) {
      tester.kh(ks).kw(1).kc(kc).xStride(257).test(
          pytorch_u8maxpool_ukernel_16x9p8q__neon);
      tester.kh(1).kw(ks).kc(kc).xStride(257).test(
          pytorch_u8maxpool_ukernel_16x9p8q__neon);
    }
  }
}

TEST(U8MAXPOOL_16x9P8Q__NEON, kc_gt_16_multipass) {
  TEST_REQUIRES_ARM_NEON;
  auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8).iterations(3);
  for (size_t ks = tester.mr() + tester.qr() + 1;
       ks < tester.mr() + 3 * tester.qr();
       ks += 3) {
    for (size_t kc = 17; kc < 32; kc++) {
      tester.kh(ks).kw(1).kc(kc).test(pytorch_u8maxpool_ukernel_16x9p8q__neon);
      tester.kh(1).kw(ks).kc(kc).test(pytorch_u8maxpool_ukernel_16x9p8q__neon);
    }
  }
}

TEST(U8MAXPOOL_16x9P8Q__NEON, kc_gt_16_multipass_with_qmin) {
  TEST_REQUIRES_ARM_NEON;
  auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8).iterations(3);
  for (size_t ks = tester.mr() + tester.qr() + 1;
       ks < tester.mr() + 3 * tester.qr();
       ks += 3) {
    for (size_t kc = 17; kc < 32; kc++) {
      tester.kh(ks).kw(1).kc(kc).qmin(192).test(
          pytorch_u8maxpool_ukernel_16x9p8q__neon);
      tester.kh(1).kw(ks).kc(kc).qmin(192).test(
          pytorch_u8maxpool_ukernel_16x9p8q__neon);
    }
  }
}

TEST(U8MAXPOOL_16x9P8Q__NEON, kc_gt_16_multipass_with_qmax) {
  TEST_REQUIRES_ARM_NEON;
  auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8).iterations(3);
  for (size_t ks = tester.mr() + tester.qr() + 1;
       ks < tester.mr() + 3 * tester.qr();
       ks += 3) {
    for (size_t kc = 17; kc < 32; kc++) {
      tester.kh(ks).kw(1).kc(kc).qmax(192).test(
          pytorch_u8maxpool_ukernel_16x9p8q__neon);
      tester.kh(1).kw(ks).kc(kc).qmax(192).test(
          pytorch_u8maxpool_ukernel_16x9p8q__neon);
    }
  }
}

TEST(U8MAXPOOL_16x9P8Q__NEON, kc_gt_16_multipass_with_x_stride) {
  TEST_REQUIRES_ARM_NEON;
  auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8).iterations(3);
  for (size_t ks = tester.mr() + tester.qr() + 1;
       ks < tester.mr() + 3 * tester.qr();
       ks += 3) {
    for (size_t kc = 17; kc < 32; kc++) {
      tester.kh(ks).kw(1).kc(kc).xStride(257).test(
          pytorch_u8maxpool_ukernel_16x9p8q__neon);
      tester.kh(1).kw(ks).kc(kc).xStride(257).test(
          pytorch_u8maxpool_ukernel_16x9p8q__neon);
    }
  }
}

TEST(U8MAXPOOL_16x9P8Q__NEON, small_n) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 2; n < 5; n++) {
    for (size_t ks : std::vector<size_t>{{2, 3, 5, 10}}) {
      for (size_t kc = 16; kc < 51; kc += 5) {
        MaxPoolMicrokernelTester()
            .kr(16)
            .mr(9)
            .qr(8)
            .n(n)
            .kh(ks)
            .kw(ks)
            .kc(kc)
            .iterations(3)
            .test(pytorch_u8maxpool_ukernel_16x9p8q__neon);
      }
    }
  }
}

TEST(U8MAXPOOL_16x9P8Q__NEON, small_n_with_x_stride) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 2; n < 5; n++) {
    for (size_t ks : std::vector<size_t>{{2, 3, 5, 10}}) {
      for (size_t kc = 16; kc < 51; kc += 5) {
        MaxPoolMicrokernelTester()
            .kr(16)
            .mr(9)
            .qr(8)
            .n(n)
            .kh(ks)
            .kw(ks)
            .kc(kc)
            .xStride(101)
            .iterations(1)
            .test(pytorch_u8maxpool_ukernel_16x9p8q__neon);
      }
    }
  }
}

TEST(U8MAXPOOL_16x9P8Q__NEON, small_n_with_y_stride) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 2; n < 5; n++) {
    for (size_t ks : std::vector<size_t>{{2, 3, 5, 10}}) {
      for (size_t kc = 16; kc < 51; kc += 5) {
        MaxPoolMicrokernelTester()
            .kr(16)
            .mr(9)
            .qr(8)
            .n(n)
            .kh(ks)
            .kw(ks)
            .kc(kc)
            .yStride(103)
            .iterations(1)
            .test(pytorch_u8maxpool_ukernel_16x9p8q__neon);
      }
    }
  }
}

TEST(U8MAXPOOL_16x9P8Q__NEON, small_n_with_s) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 2; n < 5; n++) {
    for (size_t ks : std::vector<size_t>{{2, 3, 5}}) {
      for (size_t kc = 16; kc < 51; kc += 5) {
        for (size_t s = 2; s <= ks; s++) {
          MaxPoolMicrokernelTester()
              .kr(16)
              .mr(9)
              .qr(8)
              .n(n)
              .kh(ks)
              .kw(ks)
              .kc(kc)
              .s(s)
              .iterations(1)
              .test(pytorch_u8maxpool_ukernel_16x9p8q__neon);
        }
      }
    }
  }
}
#endif

#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
TEST(U8MAXPOOL_SUB16__SSE2, kc_lt_16_mx1_pool) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t kc = 1; kc < 16; kc++) {
    for (size_t ks = 2; ks < 16; ks++) {
      MaxPoolMicrokernelTester().kr(16).kh(ks).kw(1).kc(kc).test(
          pytorch_u8maxpool_ukernel_sub16__sse2);
    }
  }
}

TEST(U8MAXPOOL_SUB16__SSE2, kc_lt_16_mx1_pool_with_qmin) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t kc = 1; kc < 16; kc++) {
    for (size_t ks = 2; ks < 16; ks++) {
      MaxPoolMicrokernelTester().kr(16).kh(ks).kw(1).kc(kc).qmin(192).test(
          pytorch_u8maxpool_ukernel_sub16__sse2);
    }
  }
}

TEST(U8MAXPOOL_SUB16__SSE2, kc_lt_16_mx1_pool_with_qmax) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t kc = 1; kc < 16; kc++) {
    for (size_t ks = 2; ks < 16; ks++) {
      MaxPoolMicrokernelTester().kr(16).kh(ks).kw(1).kc(kc).qmax(192).test(
          pytorch_u8maxpool_ukernel_sub16__sse2);
    }
  }
}

TEST(U8MAXPOOL_SUB16__SSE2, kc_lt_16_1xm_pool) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t kc = 1; kc < 16; kc++) {
    for (size_t ks = 2; ks < 16; ks++) {
      MaxPoolMicrokernelTester().kr(16).kh(1).kw(ks).kc(kc).test(
          pytorch_u8maxpool_ukernel_sub16__sse2);
    }
  }
}

TEST(U8MAXPOOL_SUB16__SSE2, kc_lt_16_1xm_pool_with_qmin) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t kc = 1; kc < 16; kc++) {
    for (size_t ks = 2; ks < 16; ks++) {
      MaxPoolMicrokernelTester().kr(16).kh(1).kw(ks).kc(kc).qmin(192).test(
          pytorch_u8maxpool_ukernel_sub16__sse2);
    }
  }
}

TEST(U8MAXPOOL_SUB16__SSE2, kc_lt_16_1xm_pool_with_qmax) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t kc = 1; kc < 16; kc++) {
    for (size_t ks = 2; ks < 16; ks++) {
      MaxPoolMicrokernelTester().kr(16).kh(1).kw(ks).kc(kc).qmax(192).test(
          pytorch_u8maxpool_ukernel_sub16__sse2);
    }
  }
}

TEST(U8MAXPOOL_SUB16__SSE2, kc_lt_16_small_n) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 2; n < 5; n++) {
    for (size_t ks : std::vector<size_t>{{2, 3, 5}}) {
      for (size_t kc = 1; kc < 16; kc++) {
        MaxPoolMicrokernelTester()
            .kr(16)
            .n(n)
            .kh(ks)
            .kw(ks)
            .kc(kc)
            .iterations(3)
            .test(pytorch_u8maxpool_ukernel_sub16__sse2);
      }
    }
  }
}

TEST(U8MAXPOOL_SUB16__SSE2, kc_lt_16_small_n_with_x_stride) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 2; n < 5; n++) {
    for (size_t ks : std::vector<size_t>{{2, 3, 5}}) {
      for (size_t kc = 1; kc < 16; kc++) {
        MaxPoolMicrokernelTester()
            .kr(16)
            .n(n)
            .kh(ks)
            .kw(ks)
            .kc(kc)
            .xStride(17)
            .iterations(3)
            .test(pytorch_u8maxpool_ukernel_sub16__sse2);
      }
    }
  }
}

TEST(U8MAXPOOL_SUB16__SSE2, kc_lt_16_small_n_with_s) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 2; n < 5; n++) {
    for (size_t ks : std::vector<size_t>{{2, 3, 5}}) {
      for (size_t s = 2; s <= 5; s++) {
        for (size_t kc = 1; kc < 16; kc++) {
          MaxPoolMicrokernelTester()
              .kr(16)
              .n(n)
              .kh(ks)
              .kw(ks)
              .kc(kc)
              .s(s)
              .iterations(1)
              .test(pytorch_u8maxpool_ukernel_sub16__sse2);
        }
      }
    }
  }
}

TEST(U8MAXPOOL_SUB16__SSE2, kc_lt_16_small_n_with_qmin) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 2; n < 5; n++) {
    for (size_t ks : std::vector<size_t>{{2, 3, 5}}) {
      for (size_t kc = 1; kc < 16; kc++) {
        MaxPoolMicrokernelTester()
            .kr(16)
            .n(n)
            .kh(ks)
            .kw(ks)
            .kc(kc)
            .qmin(192)
            .iterations(3)
            .test(pytorch_u8maxpool_ukernel_sub16__sse2);
      }
    }
  }
}

TEST(U8MAXPOOL_SUB16__SSE2, kc_lt_16_small_n_with_qmax) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 2; n < 5; n++) {
    for (size_t ks : std::vector<size_t>{{2, 3, 5}}) {
      for (size_t kc = 1; kc < 16; kc++) {
        MaxPoolMicrokernelTester()
            .kr(16)
            .n(n)
            .kh(ks)
            .kw(ks)
            .kc(kc)
            .qmax(192)
            .iterations(3)
            .test(pytorch_u8maxpool_ukernel_sub16__sse2);
      }
    }
  }
}

TEST(U8MAXPOOL_16x9P8Q__SSE2, kc_eq_16_unipass_fulltile) {
  TEST_REQUIRES_X86_SSE2;
  auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).kc(16);
  for (size_t kh = 1; kh <= tester.mr(); kh++) {
    for (size_t kw = 1; kw <= tester.mr(); kw++) {
      if (kh * kw == tester.mr()) {
        tester.kh(kh).kw(kw).test(pytorch_u8maxpool_ukernel_16x9p8q__sse2);
      }
    }
  }
}

TEST(U8MAXPOOL_16x9P8Q__SSE2, kc_eq_16_unipass_fulltile_with_qmin) {
  TEST_REQUIRES_X86_SSE2;
  auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).kc(16);
  for (size_t kh = 1; kh <= tester.mr(); kh++) {
    for (size_t kw = 1; kw <= tester.mr(); kw++) {
      if (kh * kw == tester.mr()) {
        tester.kh(kh).kw(kw).qmin(192).test(pytorch_u8maxpool_ukernel_16x9p8q__sse2);
      }
    }
  }
}

TEST(U8MAXPOOL_16x9P8Q__SSE2, kc_eq_16_unipass_fulltile_with_qmax) {
  TEST_REQUIRES_X86_SSE2;
  auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).kc(16);
  for (size_t kh = 1; kh <= tester.mr(); kh++) {
    for (size_t kw = 1; kw <= tester.mr(); kw++) {
      if (kh * kw == tester.mr()) {
        tester.kh(kh).kw(kw).qmax(192).test(pytorch_u8maxpool_ukernel_16x9p8q__sse2);
      }
    }
  }
}

TEST(U8MAXPOOL_16x9P8Q__SSE2, kc_eq_16_unipass_subtile) {
  TEST_REQUIRES_X86_SSE2;
  auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).kc(16);
  for (size_t ks = 2; ks < tester.mr(); ks++) {
    tester.kh(ks).kw(1).test(pytorch_u8maxpool_ukernel_16x9p8q__sse2);
    tester.kh(1).kw(ks).test(pytorch_u8maxpool_ukernel_16x9p8q__sse2);
  }
}

TEST(U8MAXPOOL_16x9P8Q__SSE2, kc_div_16_unipass_fulltile) {
  TEST_REQUIRES_X86_SSE2;
  auto tester = MaxPoolMicrokernelTester().kr(16).mr(9);
  for (size_t kh = 1; kh <= tester.mr(); kh++) {
    for (size_t kw = 1; kw <= tester.mr(); kw++) {
      if (kh * kw == tester.mr()) {
        for (size_t kc = 16; kc < 256; kc += 48) {
          tester.kh(kh).kw(kw).kc(kc).test(pytorch_u8maxpool_ukernel_16x9p8q__sse2);
        }
      }
    }
  }
}

TEST(U8MAXPOOL_16x9P8Q__SSE2, kc_div_16_unipass_fulltile_with_qmin) {
  TEST_REQUIRES_X86_SSE2;
  auto tester = MaxPoolMicrokernelTester().kr(16).mr(9);
  for (size_t kh = 1; kh <= tester.mr(); kh++) {
    for (size_t kw = 1; kw <= tester.mr(); kw++) {
      if (kh * kw == tester.mr()) {
        for (size_t kc = 16; kc < 256; kc += 48) {
          tester.kh(kh).kw(kw).kc(kc).qmin(192).test(
              pytorch_u8maxpool_ukernel_16x9p8q__sse2);
        }
      }
    }
  }
}

TEST(U8MAXPOOL_16x9P8Q__SSE2, kc_div_16_unipass_fulltile_with_qmax) {
  TEST_REQUIRES_X86_SSE2;
  auto tester = MaxPoolMicrokernelTester().kr(16).mr(9);
  for (size_t kh = 1; kh <= tester.mr(); kh++) {
    for (size_t kw = 1; kw <= tester.mr(); kw++) {
      if (kh * kw == tester.mr()) {
        for (size_t kc = 16; kc < 256; kc += 48) {
          tester.kh(kh).kw(kw).kc(kc).qmax(192).test(
              pytorch_u8maxpool_ukernel_16x9p8q__sse2);
        }
      }
    }
  }
}

TEST(U8MAXPOOL_16x9P8Q__SSE2, kc_div_16_unipass_fulltile_with_x_stride) {
  TEST_REQUIRES_X86_SSE2;
  auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).iterations(3);
  for (size_t kh = 1; kh <= tester.mr(); kh++) {
    for (size_t kw = 1; kw <= tester.mr(); kw++) {
      if (kh * kw == tester.mr()) {
        for (size_t kc = 16; kc < 256; kc += 48) {
          tester.kh(kh).kw(kw).kc(kc).xStride(257).test(
              pytorch_u8maxpool_ukernel_16x9p8q__sse2);
        }
      }
    }
  }
}

TEST(U8MAXPOOL_16x9P8Q__SSE2, kc_div_16_unipass_subtile) {
  TEST_REQUIRES_X86_SSE2;
  auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).iterations(3);
  for (size_t ks = 2; ks < tester.mr(); ks++) {
    for (size_t kc = 16; kc < 256; kc += 48) {
      tester.kh(ks).kw(1).kc(kc).test(pytorch_u8maxpool_ukernel_16x9p8q__sse2);
      tester.kh(1).kw(ks).kc(kc).test(pytorch_u8maxpool_ukernel_16x9p8q__sse2);
    }
  }
}

TEST(U8MAXPOOL_16x9P8Q__SSE2, kc_gt_16_unipass_fulltile) {
  TEST_REQUIRES_X86_SSE2;
  auto tester = MaxPoolMicrokernelTester().kr(16).mr(9);
  for (size_t kh = 1; kh <= tester.mr(); kh++) {
    for (size_t kw = 1; kw <= tester.mr(); kw++) {
      if (kh * kw == tester.mr()) {
        for (size_t kc = 17; kc < 32; kc++) {
          tester.kh(kh).kw(kw).kc(kc).test(pytorch_u8maxpool_ukernel_16x9p8q__sse2);
        }
      }
    }
  }
}

TEST(U8MAXPOOL_16x9P8Q__SSE2, kc_gt_16_unipass_fulltile_with_qmin) {
  TEST_REQUIRES_X86_SSE2;
  auto tester = MaxPoolMicrokernelTester().kr(16).mr(9);
  for (size_t kh = 1; kh <= tester.mr(); kh++) {
    for (size_t kw = 1; kw <= tester.mr(); kw++) {
      if (kh * kw == tester.mr()) {
        for (size_t kc = 17; kc < 32; kc++) {
          tester.kh(kh).kw(kw).kc(kc).qmin(192).test(
              pytorch_u8maxpool_ukernel_16x9p8q__sse2);
        }
      }
    }
  }
}

TEST(U8MAXPOOL_16x9P8Q__SSE2, kc_gt_16_unipass_fulltile_with_qmax) {
  TEST_REQUIRES_X86_SSE2;
  auto tester = MaxPoolMicrokernelTester().kr(16).mr(9);
  for (size_t kh = 1; kh <= tester.mr(); kh++) {
    for (size_t kw = 1; kw <= tester.mr(); kw++) {
      if (kh * kw == tester.mr()) {
        for (size_t kc = 17; kc < 32; kc++) {
          tester.kh(kh).kw(kw).kc(kc).qmax(192).test(
              pytorch_u8maxpool_ukernel_16x9p8q__sse2);
        }
      }
    }
  }
}

TEST(U8MAXPOOL_16x9P8Q__SSE2, kc_gt_16_unipass_fulltile_with_x_stride) {
  TEST_REQUIRES_X86_SSE2;
  auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).iterations(3);
  for (size_t kh = 1; kh <= tester.mr(); kh++) {
    for (size_t kw = 1; kw <= tester.mr(); kw++) {
      if (kh * kw == tester.mr()) {
        for (size_t kc = 17; kc < 32; kc++) {
          tester.kh(kh).kw(kw).kc(kc).xStride(257).test(
              pytorch_u8maxpool_ukernel_16x9p8q__sse2);
        }
      }
    }
  }
}

TEST(U8MAXPOOL_16x9P8Q__SSE2, kc_gt_16_unipass_subtile) {
  TEST_REQUIRES_X86_SSE2;
  auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).iterations(3);
  for (size_t ks = 2; ks < tester.mr(); ks++) {
    for (size_t kc = 17; kc < 32; kc++) {
      tester.kh(ks).kw(1).kc(kc).test(pytorch_u8maxpool_ukernel_16x9p8q__sse2);
      tester.kh(1).kw(ks).kc(kc).test(pytorch_u8maxpool_ukernel_16x9p8q__sse2);
    }
  }
}

TEST(U8MAXPOOL_16x9P8Q__SSE2, kc_eq_16_twopass_fulltile) {
  TEST_REQUIRES_X86_SSE2;
  auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8).kc(16);
  for (size_t kh = 1; kh <= tester.mr() + tester.qr(); kh++) {
    for (size_t kw = 1; kw <= tester.mr() + tester.qr(); kw++) {
      if (kh * kw == tester.mr() + tester.qr()) {
        tester.kh(kh).kw(kw).test(pytorch_u8maxpool_ukernel_16x9p8q__sse2);
      }
    }
  }
}

TEST(U8MAXPOOL_16x9P8Q__SSE2, kc_eq_16_twopass_fulltile_with_qmin) {
  TEST_REQUIRES_X86_SSE2;
  auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8).kc(16);
  for (size_t kh = 1; kh <= tester.mr() + tester.qr(); kh++) {
    for (size_t kw = 1; kw <= tester.mr() + tester.qr(); kw++) {
      if (kh * kw == tester.mr() + tester.qr()) {
        tester.kh(kh).kw(kw).qmin(192).test(pytorch_u8maxpool_ukernel_16x9p8q__sse2);
      }
    }
  }
}

TEST(U8MAXPOOL_16x9P8Q__SSE2, kc_eq_16_twopass_fulltile_with_qmax) {
  TEST_REQUIRES_X86_SSE2;
  auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8).kc(16);
  for (size_t kh = 1; kh <= tester.mr() + tester.qr(); kh++) {
    for (size_t kw = 1; kw <= tester.mr() + tester.qr(); kw++) {
      if (kh * kw == tester.mr() + tester.qr()) {
        tester.kh(kh).kw(kw).qmax(192).test(pytorch_u8maxpool_ukernel_16x9p8q__sse2);
      }
    }
  }
}

TEST(U8MAXPOOL_16x9P8Q__SSE2, kc_eq_16_twopass_subtile) {
  TEST_REQUIRES_X86_SSE2;
  auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8).kc(16);
  for (size_t ks = tester.mr() + 1; ks < tester.mr() + tester.qr(); ks++) {
    tester.kh(ks).kw(1).test(pytorch_u8maxpool_ukernel_16x9p8q__sse2);
    tester.kh(1).kw(ks).test(pytorch_u8maxpool_ukernel_16x9p8q__sse2);
  }
}

TEST(U8MAXPOOL_16x9P8Q__SSE2, kc_div_16_twopass_fulltile) {
  TEST_REQUIRES_X86_SSE2;
  auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8);
  for (size_t kh = 1; kh <= tester.mr() + tester.qr(); kh++) {
    for (size_t kw = 1; kw <= tester.mr() + tester.qr(); kw++) {
      if (kh * kw == tester.mr() + tester.qr()) {
        for (size_t kc = 16; kc < 256; kc += 48) {
          tester.kh(kh).kw(kw).kc(kc).test(pytorch_u8maxpool_ukernel_16x9p8q__sse2);
        }
      }
    }
  }
}

TEST(U8MAXPOOL_16x9P8Q__SSE2, kc_div_16_twopass_fulltile_with_qmin) {
  TEST_REQUIRES_X86_SSE2;
  auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8);
  for (size_t kh = 1; kh <= tester.mr() + tester.qr(); kh++) {
    for (size_t kw = 1; kw <= tester.mr() + tester.qr(); kw++) {
      if (kh * kw == tester.mr() + tester.qr()) {
        for (size_t kc = 16; kc < 256; kc += 48) {
          tester.kh(kh).kw(kw).kc(kc).qmin(192).test(
              pytorch_u8maxpool_ukernel_16x9p8q__sse2);
        }
      }
    }
  }
}

TEST(U8MAXPOOL_16x9P8Q__SSE2, kc_div_16_twopass_fulltile_with_qmax) {
  TEST_REQUIRES_X86_SSE2;
  auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8);
  for (size_t kh = 1; kh <= tester.mr() + tester.qr(); kh++) {
    for (size_t kw = 1; kw <= tester.mr() + tester.qr(); kw++) {
      if (kh * kw == tester.mr() + tester.qr()) {
        for (size_t kc = 16; kc < 256; kc += 48) {
          tester.kh(kh).kw(kw).kc(kc).qmax(192).test(
              pytorch_u8maxpool_ukernel_16x9p8q__sse2);
        }
      }
    }
  }
}

TEST(U8MAXPOOL_16x9P8Q__SSE2, kc_div_16_twopass_fulltile_with_x_stride) {
  TEST_REQUIRES_X86_SSE2;
  auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8).iterations(3);
  for (size_t kh = 1; kh <= tester.mr() + tester.qr(); kh++) {
    for (size_t kw = 1; kw <= tester.mr() + tester.qr(); kw++) {
      if (kh * kw == tester.mr() + tester.qr()) {
        for (size_t kc = 16; kc < 256; kc += 48) {
          tester.kh(kh).kw(kw).kc(kc).xStride(257).test(
              pytorch_u8maxpool_ukernel_16x9p8q__sse2);
        }
      }
    }
  }
}

TEST(U8MAXPOOL_16x9P8Q__SSE2, kc_div_16_twopass_subtile) {
  TEST_REQUIRES_X86_SSE2;
  auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8).iterations(3);
  for (size_t ks = tester.mr() + 1; ks < tester.mr() + tester.qr(); ks++) {
    for (size_t kc = 16; kc < 256; kc += 48) {
      tester.kh(ks).kw(1).kc(kc).test(pytorch_u8maxpool_ukernel_16x9p8q__sse2);
      tester.kh(1).kw(ks).kc(kc).test(pytorch_u8maxpool_ukernel_16x9p8q__sse2);
    }
  }
}

TEST(U8MAXPOOL_16x9P8Q__SSE2, kc_gt_16_twopass_fulltile) {
  TEST_REQUIRES_X86_SSE2;
  auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8);
  for (size_t kh = 1; kh <= tester.mr() + tester.qr(); kh++) {
    for (size_t kw = 1; kw <= tester.mr() + tester.qr(); kw++) {
      if (kh * kw == tester.mr() + tester.qr()) {
        for (size_t kc = 17; kc < 32; kc++) {
          tester.kh(kh).kw(kw).kc(kc).test(pytorch_u8maxpool_ukernel_16x9p8q__sse2);
        }
      }
    }
  }
}

TEST(U8MAXPOOL_16x9P8Q__SSE2, kc_gt_16_twopass_fulltile_with_qmin) {
  TEST_REQUIRES_X86_SSE2;
  auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8);
  for (size_t kh = 1; kh <= tester.mr() + tester.qr(); kh++) {
    for (size_t kw = 1; kw <= tester.mr() + tester.qr(); kw++) {
      if (kh * kw == tester.mr() + tester.qr()) {
        for (size_t kc = 17; kc < 32; kc++) {
          tester.kh(kh).kw(kw).kc(kc).qmin(192).test(
              pytorch_u8maxpool_ukernel_16x9p8q__sse2);
        }
      }
    }
  }
}

TEST(U8MAXPOOL_16x9P8Q__SSE2, kc_gt_16_twopass_fulltile_with_qmax) {
  TEST_REQUIRES_X86_SSE2;
  auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8);
  for (size_t kh = 1; kh <= tester.mr() + tester.qr(); kh++) {
    for (size_t kw = 1; kw <= tester.mr() + tester.qr(); kw++) {
      if (kh * kw == tester.mr() + tester.qr()) {
        for (size_t kc = 17; kc < 32; kc++) {
          tester.kh(kh).kw(kw).kc(kc).qmax(192).test(
              pytorch_u8maxpool_ukernel_16x9p8q__sse2);
        }
      }
    }
  }
}

TEST(U8MAXPOOL_16x9P8Q__SSE2, kc_gt_16_twopass_fulltile_with_x_stride) {
  TEST_REQUIRES_X86_SSE2;
  auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8).iterations(3);
  for (size_t kh = 1; kh <= tester.mr() + tester.qr(); kh++) {
    for (size_t kw = 1; kw <= tester.mr() + tester.qr(); kw++) {
      if (kh * kw == tester.mr() + tester.qr()) {
        for (size_t kc = 17; kc < 32; kc++) {
          tester.kh(kh).kw(kw).kc(kc).xStride(257).test(
              pytorch_u8maxpool_ukernel_16x9p8q__sse2);
        }
      }
    }
  }
}

TEST(U8MAXPOOL_16x9P8Q__SSE2, kc_gt_16_twopass_subtile) {
  TEST_REQUIRES_X86_SSE2;
  auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8).iterations(3);
  for (size_t ks = tester.mr() + 1; ks < tester.mr() + tester.qr(); ks++) {
    for (size_t kc = 17; kc < 32; kc++) {
      tester.kh(ks).kw(1).kc(kc).test(pytorch_u8maxpool_ukernel_16x9p8q__sse2);
      tester.kh(1).kw(ks).kc(kc).test(pytorch_u8maxpool_ukernel_16x9p8q__sse2);
    }
  }
}

TEST(U8MAXPOOL_16x9P8Q__SSE2, kc_eq_16_multipass) {
  TEST_REQUIRES_X86_SSE2;
  auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8).kc(16);
  for (size_t ks = tester.mr() + tester.qr() + 1;
       ks < tester.mr() + 3 * tester.qr();
       ks += 3) {
    tester.kh(ks).kw(1).test(pytorch_u8maxpool_ukernel_16x9p8q__sse2);
    tester.kh(1).kw(ks).test(pytorch_u8maxpool_ukernel_16x9p8q__sse2);
  }
}

TEST(U8MAXPOOL_16x9P8Q__SSE2, kc_eq_16_multipass_with_qmin) {
  TEST_REQUIRES_X86_SSE2;
  auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8).kc(16);
  for (size_t ks = tester.mr() + tester.qr() + 1;
       ks < tester.mr() + 3 * tester.qr();
       ks += 3) {
    tester.kh(ks).kw(1).qmin(192).test(pytorch_u8maxpool_ukernel_16x9p8q__sse2);
    tester.kh(1).kw(ks).qmin(192).test(pytorch_u8maxpool_ukernel_16x9p8q__sse2);
  }
}

TEST(U8MAXPOOL_16x9P8Q__SSE2, kc_eq_16_multipass_with_qmax) {
  TEST_REQUIRES_X86_SSE2;
  auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8).kc(16);
  for (size_t ks = tester.mr() + tester.qr() + 1;
       ks < tester.mr() + 3 * tester.qr();
       ks += 3) {
    tester.kh(ks).kw(1).qmax(192).test(pytorch_u8maxpool_ukernel_16x9p8q__sse2);
    tester.kh(1).kw(ks).qmax(192).test(pytorch_u8maxpool_ukernel_16x9p8q__sse2);
  }
}

TEST(U8MAXPOOL_16x9P8Q__SSE2, kc_div_16_multipass) {
  TEST_REQUIRES_X86_SSE2;
  auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8);
  for (size_t ks = tester.mr() + tester.qr() + 1;
       ks < tester.mr() + 3 * tester.qr();
       ks += 3) {
    for (size_t kc = 16; kc < 256; kc += 48) {
      tester.kh(ks).kw(1).kc(kc).test(pytorch_u8maxpool_ukernel_16x9p8q__sse2);
      tester.kh(1).kw(ks).kc(kc).test(pytorch_u8maxpool_ukernel_16x9p8q__sse2);
    }
  }
}

TEST(U8MAXPOOL_16x9P8Q__SSE2, kc_div_16_multipass_with_qmin) {
  TEST_REQUIRES_X86_SSE2;
  auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8);
  for (size_t ks = tester.mr() + tester.qr() + 1;
       ks < tester.mr() + 3 * tester.qr();
       ks += 3) {
    for (size_t kc = 16; kc < 256; kc += 48) {
      tester.kh(ks).kw(1).kc(kc).qmin(192).test(
          pytorch_u8maxpool_ukernel_16x9p8q__sse2);
      tester.kh(1).kw(ks).kc(kc).qmin(192).test(
          pytorch_u8maxpool_ukernel_16x9p8q__sse2);
    }
  }
}

TEST(U8MAXPOOL_16x9P8Q__SSE2, kc_div_16_multipass_with_qmax) {
  TEST_REQUIRES_X86_SSE2;
  auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8);
  for (size_t ks = tester.mr() + tester.qr() + 1;
       ks < tester.mr() + 3 * tester.qr();
       ks += 3) {
    for (size_t kc = 16; kc < 256; kc += 48) {
      tester.kh(ks).kw(1).kc(kc).qmax(192).test(
          pytorch_u8maxpool_ukernel_16x9p8q__sse2);
      tester.kh(1).kw(ks).kc(kc).qmax(192).test(
          pytorch_u8maxpool_ukernel_16x9p8q__sse2);
    }
  }
}

TEST(U8MAXPOOL_16x9P8Q__SSE2, kc_div_16_multipass_with_x_stride) {
  TEST_REQUIRES_X86_SSE2;
  auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8).iterations(3);
  for (size_t ks = tester.mr() + tester.qr() + 1;
       ks < tester.mr() + 3 * tester.qr();
       ks += 3) {
    for (size_t kc = 16; kc < 256; kc += 48) {
      tester.kh(ks).kw(1).kc(kc).xStride(257).test(
          pytorch_u8maxpool_ukernel_16x9p8q__sse2);
      tester.kh(1).kw(ks).kc(kc).xStride(257).test(
          pytorch_u8maxpool_ukernel_16x9p8q__sse2);
    }
  }
}

TEST(U8MAXPOOL_16x9P8Q__SSE2, kc_gt_16_multipass) {
  TEST_REQUIRES_X86_SSE2;
  auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8).iterations(3);
  for (size_t ks = tester.mr() + tester.qr() + 1;
       ks < tester.mr() + 3 * tester.qr();
       ks += 3) {
    for (size_t kc = 17; kc < 32; kc++) {
      tester.kh(ks).kw(1).kc(kc).test(pytorch_u8maxpool_ukernel_16x9p8q__sse2);
      tester.kh(1).kw(ks).kc(kc).test(pytorch_u8maxpool_ukernel_16x9p8q__sse2);
    }
  }
}

TEST(U8MAXPOOL_16x9P8Q__SSE2, kc_gt_16_multipass_with_qmin) {
  TEST_REQUIRES_X86_SSE2;
  auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8).iterations(3);
  for (size_t ks = tester.mr() + tester.qr() + 1;
       ks < tester.mr() + 3 * tester.qr();
       ks += 3) {
    for (size_t kc = 17; kc < 32; kc++) {
      tester.kh(ks).kw(1).kc(kc).qmin(192).test(
          pytorch_u8maxpool_ukernel_16x9p8q__sse2);
      tester.kh(1).kw(ks).kc(kc).qmin(192).test(
          pytorch_u8maxpool_ukernel_16x9p8q__sse2);
    }
  }
}

TEST(U8MAXPOOL_16x9P8Q__SSE2, kc_gt_16_multipass_with_qmax) {
  TEST_REQUIRES_X86_SSE2;
  auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8).iterations(3);
  for (size_t ks = tester.mr() + tester.qr() + 1;
       ks < tester.mr() + 3 * tester.qr();
       ks += 3) {
    for (size_t kc = 17; kc < 32; kc++) {
      tester.kh(ks).kw(1).kc(kc).qmax(192).test(
          pytorch_u8maxpool_ukernel_16x9p8q__sse2);
      tester.kh(1).kw(ks).kc(kc).qmax(192).test(
          pytorch_u8maxpool_ukernel_16x9p8q__sse2);
    }
  }
}

TEST(U8MAXPOOL_16x9P8Q__SSE2, kc_gt_16_multipass_with_x_stride) {
  TEST_REQUIRES_X86_SSE2;
  auto tester = MaxPoolMicrokernelTester().kr(16).mr(9).qr(8).iterations(3);
  for (size_t ks = tester.mr() + tester.qr() + 1;
       ks < tester.mr() + 3 * tester.qr();
       ks += 3) {
    for (size_t kc = 17; kc < 32; kc++) {
      tester.kh(ks).kw(1).kc(kc).xStride(257).test(
          pytorch_u8maxpool_ukernel_16x9p8q__sse2);
      tester.kh(1).kw(ks).kc(kc).xStride(257).test(
          pytorch_u8maxpool_ukernel_16x9p8q__sse2);
    }
  }
}

TEST(U8MAXPOOL_16x9P8Q__SSE2, small_n) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 2; n < 5; n++) {
    for (size_t ks : std::vector<size_t>{{2, 3, 5, 10}}) {
      for (size_t kc = 16; kc < 51; kc += 5) {
        MaxPoolMicrokernelTester()
            .kr(16)
            .mr(9)
            .qr(8)
            .n(n)
            .kh(ks)
            .kw(ks)
            .kc(kc)
            .iterations(3)
            .test(pytorch_u8maxpool_ukernel_16x9p8q__sse2);
      }
    }
  }
}

TEST(U8MAXPOOL_16x9P8Q__SSE2, small_n_with_x_stride) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 2; n < 5; n++) {
    for (size_t ks : std::vector<size_t>{{2, 3, 5, 10}}) {
      for (size_t kc = 16; kc < 51; kc += 5) {
        MaxPoolMicrokernelTester()
            .kr(16)
            .mr(9)
            .qr(8)
            .n(n)
            .kh(ks)
            .kw(ks)
            .kc(kc)
            .xStride(101)
            .iterations(1)
            .test(pytorch_u8maxpool_ukernel_16x9p8q__sse2);
      }
    }
  }
}

TEST(U8MAXPOOL_16x9P8Q__SSE2, small_n_with_y_stride) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 2; n < 5; n++) {
    for (size_t ks : std::vector<size_t>{{2, 3, 5, 10}}) {
      for (size_t kc = 16; kc < 51; kc += 5) {
        MaxPoolMicrokernelTester()
            .kr(16)
            .mr(9)
            .qr(8)
            .n(n)
            .kh(ks)
            .kw(ks)
            .kc(kc)
            .yStride(103)
            .iterations(1)
            .test(pytorch_u8maxpool_ukernel_16x9p8q__sse2);
      }
    }
  }
}

TEST(U8MAXPOOL_16x9P8Q__SSE2, small_n_with_s) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 2; n < 5; n++) {
    for (size_t ks : std::vector<size_t>{{2, 3, 5}}) {
      for (size_t kc = 16; kc < 51; kc += 5) {
        for (size_t s = 2; s <= ks; s++) {
          MaxPoolMicrokernelTester()
              .kr(16)
              .mr(9)
              .qr(8)
              .n(n)
              .kh(ks)
              .kw(ks)
              .kc(kc)
              .s(s)
              .iterations(1)
              .test(pytorch_u8maxpool_ukernel_16x9p8q__sse2);
        }
      }
    }
  }
}
#endif /* CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64 */
