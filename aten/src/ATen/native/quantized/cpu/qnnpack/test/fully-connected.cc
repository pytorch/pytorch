/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "fully-connected-operator-tester.h"

TEST(FULLY_CONNECTED_OP, integration_test) {
  FullyConnectedOperatorTester()
      .batchSize(4)
      .inputChannels(4)
      .outputChannels(4)
      .iterations(3)
      .testQ8();
}

TEST(FULLY_CONNECTED_OP, zero_batch) {
  FullyConnectedOperatorTester()
      .batchSize(0)
      .inputChannels(2)
      .outputChannels(2)
      .iterations(1)
      .testQ8();
}

TEST(FULLY_CONNECTED_OP, unit_batch) {
  FullyConnectedOperatorTester()
      .batchSize(1)
      .inputChannels(23)
      .outputChannels(19)
      .iterations(3)
      .testQ8();
}

TEST(FULLY_CONNECTED_OP, unit_batch_with_qmin) {
  FullyConnectedOperatorTester()
      .batchSize(1)
      .inputChannels(23)
      .outputChannels(19)
      .qmin(128)
      .iterations(3)
      .testQ8();
}

TEST(FULLY_CONNECTED_OP, unit_batch_with_qmax) {
  FullyConnectedOperatorTester()
      .batchSize(1)
      .inputChannels(23)
      .outputChannels(19)
      .qmax(128)
      .iterations(3)
      .testQ8();
}

TEST(FULLY_CONNECTED_OP, unit_batch_with_input_stride) {
  FullyConnectedOperatorTester()
      .batchSize(1)
      .inputChannels(23)
      .inputStride(28)
      .outputChannels(19)
      .iterations(3)
      .testQ8();
}

TEST(FULLY_CONNECTED_OP, unit_batch_with_output_stride) {
  FullyConnectedOperatorTester()
      .batchSize(1)
      .inputChannels(23)
      .outputChannels(19)
      .outputStride(29)
      .iterations(3)
      .testQ8();
}

TEST(FULLY_CONNECTED_OP, small_batch) {
  FullyConnectedOperatorTester()
      .batchSize(12)
      .inputChannels(23)
      .outputChannels(19)
      .iterations(3)
      .testQ8();
}

TEST(FULLY_CONNECTED_OP, small_batch_with_qmin) {
  FullyConnectedOperatorTester()
      .batchSize(12)
      .inputChannels(23)
      .outputChannels(19)
      .qmin(128)
      .iterations(3)
      .testQ8();
}

TEST(FULLY_CONNECTED_OP, small_batch_with_qmax) {
  FullyConnectedOperatorTester()
      .batchSize(12)
      .inputChannels(23)
      .outputChannels(19)
      .qmax(128)
      .iterations(3)
      .testQ8();
}

TEST(FULLY_CONNECTED_OP, small_batch_with_input_stride) {
  FullyConnectedOperatorTester()
      .batchSize(12)
      .inputChannels(23)
      .inputStride(28)
      .outputChannels(19)
      .iterations(3)
      .testQ8();
}

TEST(FULLY_CONNECTED_OP, small_batch_with_output_stride) {
  FullyConnectedOperatorTester()
      .batchSize(12)
      .inputChannels(23)
      .outputChannels(19)
      .outputStride(29)
      .iterations(3)
      .testQ8();
}

TEST(FULLY_CONNECTED_OP, runtime_quant) {
  FullyConnectedOperatorTester()
      .batchSize(4)
      .inputChannels(4)
      .outputChannels(4)
      .iterations(3)
      .testQ8(true);
}
