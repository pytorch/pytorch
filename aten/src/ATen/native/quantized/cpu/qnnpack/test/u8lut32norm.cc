/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <qnnpack/u8lut32norm.h>

#include "lut-norm-microkernel-tester.h"

TEST(U8LUT32NORM__SCALAR, n_eq_1) {
  LUTNormMicrokernelTester().n(1).test(pytorch_u8lut32norm_ukernel__scalar);
}

TEST(U8LUT32NORM__SCALAR, small_n) {
  for (size_t n = 2; n <= 16; n++) {
    LUTNormMicrokernelTester().n(n).test(pytorch_u8lut32norm_ukernel__scalar);
  }
}

TEST(U8LUT32NORM__SCALAR, large_n) {
  for (size_t n = 16; n <= 128; n += 2) {
    LUTNormMicrokernelTester().n(n).test(pytorch_u8lut32norm_ukernel__scalar);
  }
}

TEST(U8LUT32NORM__SCALAR, n_eq_1_inplace) {
  LUTNormMicrokernelTester().n(1).inplace(true).test(
      pytorch_u8lut32norm_ukernel__scalar);
}

TEST(U8LUT32NORM__SCALAR, small_n_inplace) {
  for (size_t n = 2; n <= 16; n++) {
    LUTNormMicrokernelTester().n(n).inplace(true).test(
        pytorch_u8lut32norm_ukernel__scalar);
  }
}

TEST(U8LUT32NORM__SCALAR, large_n_inplace) {
  for (size_t n = 16; n <= 128; n += 2) {
    LUTNormMicrokernelTester().n(n).inplace(true).test(
        pytorch_u8lut32norm_ukernel__scalar);
  }
}
