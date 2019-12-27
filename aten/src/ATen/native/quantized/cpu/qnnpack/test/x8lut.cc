/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <qnnpack/x8lut.h>

#include "lut-microkernel-tester.h"

TEST(X8LUT__SCALAR, n_eq_1) {
  LUTMicrokernelTester().n(1).test(pytorch_x8lut_ukernel__scalar);
}

TEST(X8LUT__SCALAR, small_n) {
  for (size_t n = 2; n <= 16; n++) {
    LUTMicrokernelTester().n(n).test(pytorch_x8lut_ukernel__scalar);
  }
}

TEST(X8LUT__SCALAR, large_n) {
  for (size_t n = 16; n <= 128; n += 2) {
    LUTMicrokernelTester().n(n).test(pytorch_x8lut_ukernel__scalar);
  }
}

TEST(X8LUT__SCALAR, n_eq_1_inplace) {
  LUTMicrokernelTester().n(1).inplace(true).test(pytorch_x8lut_ukernel__scalar);
}

TEST(X8LUT__SCALAR, small_n_inplace) {
  for (size_t n = 2; n <= 16; n++) {
    LUTMicrokernelTester().n(n).inplace(true).test(pytorch_x8lut_ukernel__scalar);
  }
}

TEST(X8LUT__SCALAR, large_n_inplace) {
  for (size_t n = 16; n <= 128; n += 2) {
    LUTMicrokernelTester().n(n).inplace(true).test(pytorch_x8lut_ukernel__scalar);
  }
}
