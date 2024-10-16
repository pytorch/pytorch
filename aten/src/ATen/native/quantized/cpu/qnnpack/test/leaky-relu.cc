/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "leaky-relu-operator-tester.h"

TEST(LEAKY_RELU_OP, zero_batch) {
  LeakyReLUOperatorTester().batchSize(0).channels(2).iterations(1).testQ8();
}

TEST(LEAKY_RELU_OP, unit_batch) {
  for (size_t channels = 1; channels < 100; channels++) {
    LeakyReLUOperatorTester()
        .batchSize(1)
        .channels(channels)
        .iterations(3)
        .testQ8();
  }
}

TEST(LEAKY_RELU_OP, unit_batch_with_qmin) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    LeakyReLUOperatorTester()
        .batchSize(1)
        .channels(channels)
        .qmin(128)
        .iterations(3)
        .testQ8();
  }
}

TEST(LEAKY_RELU_OP, unit_batch_with_qmax) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    LeakyReLUOperatorTester()
        .batchSize(1)
        .channels(channels)
        .qmax(128)
        .iterations(3)
        .testQ8();
  }
}

TEST(LEAKY_RELU_OP, unit_batch_with_negative_slope) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (float negativeSlope = 1.0e-4f; negativeSlope < 1.0f;
         negativeSlope *= 3.14159265f) {
      LeakyReLUOperatorTester()
          .batchSize(1)
          .channels(channels)
          .negativeSlope(negativeSlope)
          .iterations(1)
          .testQ8();
    }
  }
}

TEST(LEAKY_RELU_OP, unit_batch_with_input_scale) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (float inputScale = 1.0e-2f; inputScale < 1.0e+2f;
         inputScale *= 3.14159265f) {
      LeakyReLUOperatorTester()
          .batchSize(1)
          .channels(channels)
          .inputScale(inputScale)
          .iterations(1)
          .testQ8();
    }
  }
}

TEST(LEAKY_RELU_OP, unit_batch_with_input_zero_point) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (int32_t inputZeroPoint = 0; inputZeroPoint <= 255;
         inputZeroPoint += 51) {
      LeakyReLUOperatorTester()
          .batchSize(1)
          .channels(channels)
          .inputZeroPoint(uint8_t(inputZeroPoint))
          .iterations(1)
          .testQ8();
    }
  }
}

TEST(LEAKY_RELU_OP, unit_batch_with_output_scale) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (float outputScale = 1.0e-2f; outputScale < 1.0e+2f;
         outputScale *= 3.14159265f) {
      LeakyReLUOperatorTester()
          .batchSize(1)
          .channels(channels)
          .outputScale(outputScale)
          .iterations(1)
          .testQ8();
    }
  }
}

TEST(LEAKY_RELU_OP, unit_batch_with_output_zero_point) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (int32_t outputZeroPoint = 0; outputZeroPoint <= 255;
         outputZeroPoint += 51) {
      LeakyReLUOperatorTester()
          .batchSize(1)
          .channels(channels)
          .outputZeroPoint(uint8_t(outputZeroPoint))
          .iterations(1)
          .testQ8();
    }
  }
}

TEST(LEAKY_RELU_OP, small_batch) {
  for (size_t channels = 1; channels < 100; channels++) {
    LeakyReLUOperatorTester()
        .batchSize(3)
        .channels(channels)
        .iterations(3)
        .testQ8();
  }
}

TEST(LEAKY_RELU_OP, small_batch_with_input_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    LeakyReLUOperatorTester()
        .batchSize(3)
        .channels(channels)
        .inputStride(129)
        .iterations(3)
        .testQ8();
  }
}

TEST(LEAKY_RELU_OP, small_batch_with_output_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    LeakyReLUOperatorTester()
        .batchSize(3)
        .channels(channels)
        .outputStride(117)
        .iterations(3)
        .testQ8();
  }
}

TEST(LEAKY_RELU_OP, small_batch_with_input_and_output_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    LeakyReLUOperatorTester()
        .batchSize(3)
        .channels(channels)
        .inputStride(129)
        .outputStride(117)
        .iterations(3)
        .testQ8();
  }
}
