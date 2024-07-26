/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "softargmax-operator-tester.h"

#include <qnnpack/params.h>

TEST(SOFTARGMAX_OP, zero_batch) {
  SoftArgMaxOperatorTester().batchSize(0).channels(1).iterations(1).testQ8();
}

TEST(SOFTARGMAX_OP, single_class) {
  SoftArgMaxOperatorTester().batchSize(1).channels(1).iterations(100).testQ8();
}

TEST(SOFTARGMAX_OP, two_classes) {
  SoftArgMaxOperatorTester().batchSize(1).channels(2).iterations(100).testQ8();
}

TEST(SOFTARGMAX_OP, many_classes) {
  for (size_t channels = 3; channels < 100; channels++) {
    SoftArgMaxOperatorTester()
        .batchSize(1)
        .channels(channels)
        .iterations(1)
        .testQ8();
  }
}

TEST(SOFTARGMAX_OP, cifar_classes) {
  /* CIFAR-10 */
  SoftArgMaxOperatorTester().batchSize(1).channels(10).iterations(15).testQ8();
  /* CIFAR-100 */
  SoftArgMaxOperatorTester().batchSize(1).channels(100).iterations(15).testQ8();
}

TEST(SOFTARGMAX_OP, imagenet_classes) {
  /* ImageNet-1K */
  SoftArgMaxOperatorTester()
      .batchSize(1)
      .channels(1000)
      .iterations(10)
      .testQ8();
  /* ImageNet-1K+1 */
  SoftArgMaxOperatorTester()
      .batchSize(1)
      .channels(1001)
      .iterations(10)
      .testQ8();
  /* ImageNet-22K */
  SoftArgMaxOperatorTester()
      .batchSize(1)
      .channels(21841)
      .iterations(10)
      .testQ8();
}

TEST(SOFTARGMAX_OP, many_channels_with_input_scale) {
  for (size_t channels = 1; channels < 100; channels += 5) {
    for (float inputScale = 1.0e-2f; inputScale < 1.0e+2f;
         inputScale *= 3.14159265f) {
      SoftArgMaxOperatorTester()
          .batchSize(1)
          .channels(channels)
          .inputScale(inputScale)
          .iterations(1)
          .testQ8();
    }
  }
}

TEST(SOFTARGMAX_OP, many_channels_with_input_zero_point) {
  for (size_t channels = 1; channels < 100; channels += 5) {
    for (int32_t inputZeroPoint = 0; inputZeroPoint <= 255;
         inputZeroPoint += 51) {
      SoftArgMaxOperatorTester()
          .batchSize(1)
          .channels(channels)
          .inputZeroPoint(uint8_t(inputZeroPoint))
          .iterations(1)
          .testQ8();
    }
  }
}

TEST(SOFTARGMAX_OP, small_batch) {
  for (size_t channels = 1; channels < 100; channels += 5) {
    SoftArgMaxOperatorTester()
        .batchSize(3)
        .channels(channels)
        .iterations(3)
        .testQ8();
  }
}

TEST(SOFTARGMAX_OP, small_batch_with_input_stride) {
  for (size_t channels = 1; channels < 100; channels += 5) {
    SoftArgMaxOperatorTester()
        .batchSize(3)
        .channels(channels)
        .inputStride(129)
        .iterations(3)
        .testQ8();
  }
}

TEST(SOFTARGMAX_OP, small_batch_with_output_stride) {
  for (size_t channels = 1; channels < 100; channels += 5) {
    SoftArgMaxOperatorTester()
        .batchSize(3)
        .channels(channels)
        .outputStride(117)
        .iterations(3)
        .testQ8();
  }
}

TEST(SOFTARGMAX_OP, strided_batch_with_input_and_output_stride) {
  for (size_t channels = 1; channels < 100; channels += 5) {
    SoftArgMaxOperatorTester()
        .batchSize(3)
        .channels(channels)
        .inputStride(129)
        .outputStride(117)
        .iterations(3)
        .testQ8();
  }
}
