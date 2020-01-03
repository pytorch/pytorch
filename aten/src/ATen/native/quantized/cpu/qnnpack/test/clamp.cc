/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "clamp-operator-tester.h"

TEST(CLAMP_OP, zero_batch) {
  ClampOperatorTester().batchSize(0).channels(2).iterations(1).testU8();
}

TEST(CLAMP_OP, unit_batch) {
  for (size_t channels = 1; channels < 100; channels++) {
    ClampOperatorTester()
        .batchSize(1)
        .channels(channels)
        .iterations(3)
        .testU8();
  }
}

TEST(CLAMP_OP, unit_batch_with_qmin) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (uint8_t qmin = 1; qmin < 255; qmin++) {
      ClampOperatorTester()
          .batchSize(1)
          .channels(channels)
          .qmin(qmin)
          .iterations(3)
          .testU8();
    }
  }
}

TEST(CLAMP_OP, unit_batch_with_qmax) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (uint8_t qmax = 1; qmax < 255; qmax++) {
      ClampOperatorTester()
          .batchSize(1)
          .channels(channels)
          .qmax(qmax)
          .iterations(3)
          .testU8();
    }
  }
}

TEST(CLAMP_OP, small_batch) {
  for (size_t channels = 1; channels < 100; channels++) {
    ClampOperatorTester()
        .batchSize(3)
        .channels(channels)
        .iterations(3)
        .testU8();
  }
}

TEST(CLAMP_OP, small_batch_with_input_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    ClampOperatorTester()
        .batchSize(3)
        .channels(channels)
        .inputStride(129)
        .iterations(3)
        .testU8();
  }
}

TEST(CLAMP_OP, small_batch_with_output_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    ClampOperatorTester()
        .batchSize(3)
        .channels(channels)
        .outputStride(117)
        .iterations(3)
        .testU8();
  }
}

TEST(CLAMP_OP, small_batch_with_input_and_output_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    ClampOperatorTester()
        .batchSize(3)
        .channels(channels)
        .inputStride(129)
        .outputStride(117)
        .iterations(3)
        .testU8();
  }
}

TEST(CLAMP_OP, qmin_and_qmax_equal_uint8_max) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    ClampOperatorTester()
        .batchSize(3)
        .channels(channels)
        .qmin(255)
        .qmax(255)
        .iterations(3)
        .testU8();
  }
}
