/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "fully-connected-sparse-operator-tester.h"

TEST(FULLY_CONNECTED_SPARSE_OP, integration_test_dynamic) {
  FullyConnectedSparseOperatorTester()
      .batchSize(4)
      .inputChannels(4)
      .outputChannels(4)
      .iterations(3)
      .testQ8(FullyConnectedSparseOperatorTester::Mode::Dynamic);
}

TEST(FULLY_CONNECTED_SPARSE_OP, zero_batch_dynamic) {
  FullyConnectedSparseOperatorTester()
      .batchSize(0)
      .inputChannels(2)
      .outputChannels(2)
      .iterations(1)
      .testQ8(FullyConnectedSparseOperatorTester::Mode::Dynamic);
}

TEST(FULLY_CONNECTED_SPARSE_OP, unit_batch_dynamic) {
  FullyConnectedSparseOperatorTester()
      .batchSize(1)
      .inputChannels(23)
      .outputChannels(19)
      .iterations(3)
      .testQ8(FullyConnectedSparseOperatorTester::Mode::Dynamic);
}

TEST(FULLY_CONNECTED_SPARSE_OP, unit_batch_with_qmin_dynamic) {
  FullyConnectedSparseOperatorTester()
      .batchSize(1)
      .inputChannels(23)
      .outputChannels(19)
      .qmin(128)
      .iterations(3)
      .testQ8(FullyConnectedSparseOperatorTester::Mode::Dynamic);
}

TEST(FULLY_CONNECTED_SPARSE_OP, unit_batch_with_qmax_dynamic) {
  FullyConnectedSparseOperatorTester()
      .batchSize(1)
      .inputChannels(23)
      .outputChannels(19)
      .qmax(128)
      .iterations(3)
      .testQ8(FullyConnectedSparseOperatorTester::Mode::Dynamic);
}

TEST(FULLY_CONNECTED_SPARSE_OP, unit_batch_with_input_stride_dynamic) {
  FullyConnectedSparseOperatorTester()
      .batchSize(1)
      .inputChannels(23)
      .inputStride(28)
      .outputChannels(19)
      .iterations(3)
      .testQ8(FullyConnectedSparseOperatorTester::Mode::Dynamic);
}

TEST(FULLY_CONNECTED_SPARSE_OP, unit_batch_with_output_stride_dynamic) {
  FullyConnectedSparseOperatorTester()
      .batchSize(1)
      .inputChannels(23)
      .outputChannels(19)
      .outputStride(29)
      .iterations(3)
      .testQ8(FullyConnectedSparseOperatorTester::Mode::Dynamic);
}

TEST(FULLY_CONNECTED_SPARSE_OP, small_batch_dynamic) {
  FullyConnectedSparseOperatorTester()
      .batchSize(12)
      .inputChannels(23)
      .outputChannels(19)
      .iterations(3)
      .testQ8(FullyConnectedSparseOperatorTester::Mode::Dynamic);
}

TEST(FULLY_CONNECTED_SPARSE_OP, small_batch_with_qmin_dynamic) {
  FullyConnectedSparseOperatorTester()
      .batchSize(12)
      .inputChannels(23)
      .outputChannels(19)
      .qmin(128)
      .iterations(3)
      .testQ8(FullyConnectedSparseOperatorTester::Mode::Dynamic);
}

TEST(FULLY_CONNECTED_SPARSE_OP, small_batch_with_qmax_dynamic) {
  FullyConnectedSparseOperatorTester()
      .batchSize(12)
      .inputChannels(23)
      .outputChannels(19)
      .qmax(128)
      .iterations(3)
      .testQ8(FullyConnectedSparseOperatorTester::Mode::Dynamic);
}

TEST(FULLY_CONNECTED_OP, integration_test_dynamic_prepacked) {
  FullyConnectedSparseOperatorTester()
      .batchSize(4)
      .inputChannels(4)
      .outputChannels(4)
      .iterations(3)
      .testQ8_prepacked(FullyConnectedSparseOperatorTester::Mode::Dynamic);
}

TEST(FULLY_CONNECTED_OP, zero_batch_dynamic_prepacked) {
  FullyConnectedSparseOperatorTester()
      .batchSize(0)
      .inputChannels(2)
      .outputChannels(2)
      .iterations(1)
      .testQ8_prepacked(FullyConnectedSparseOperatorTester::Mode::Dynamic);
}

TEST(FULLY_CONNECTED_OP, unit_batch_dynamic_prepacked) {
  FullyConnectedSparseOperatorTester()
      .batchSize(1)
      .inputChannels(23)
      .outputChannels(19)
      .iterations(3)
      .testQ8_prepacked(FullyConnectedSparseOperatorTester::Mode::Dynamic);
}

TEST(FULLY_CONNECTED_OP, unit_batch_with_qmin_dynamic_prepacked) {
  FullyConnectedSparseOperatorTester()
      .batchSize(1)
      .inputChannels(23)
      .outputChannels(19)
      .qmin(128)
      .iterations(3)
      .testQ8_prepacked(FullyConnectedSparseOperatorTester::Mode::Dynamic);
}

TEST(FULLY_CONNECTED_OP, unit_batch_with_qmax_dynamic_prepacked) {
  FullyConnectedSparseOperatorTester()
      .batchSize(1)
      .inputChannels(23)
      .outputChannels(19)
      .qmax(128)
      .iterations(3)
      .testQ8_prepacked(FullyConnectedSparseOperatorTester::Mode::Dynamic);
}

TEST(FULLY_CONNECTED_OP, unit_batch_with_input_stride_dynamic_prepacked) {
  FullyConnectedSparseOperatorTester()
      .batchSize(1)
      .inputChannels(23)
      .inputStride(28)
      .outputChannels(19)
      .iterations(3)
      .testQ8_prepacked(FullyConnectedSparseOperatorTester::Mode::Dynamic);
}

TEST(FULLY_CONNECTED_OP, unit_batch_with_output_stride_dynamic_prepacked) {
  FullyConnectedSparseOperatorTester()
      .batchSize(1)
      .inputChannels(23)
      .outputChannels(19)
      .outputStride(29)
      .iterations(3)
      .testQ8_prepacked(FullyConnectedSparseOperatorTester::Mode::Dynamic);
}

TEST(FULLY_CONNECTED_OP, small_batch_dynamic_prepacked) {
  FullyConnectedSparseOperatorTester()
      .batchSize(12)
      .inputChannels(23)
      .outputChannels(19)
      .iterations(3)
      .testQ8_prepacked(FullyConnectedSparseOperatorTester::Mode::Dynamic);
}

TEST(FULLY_CONNECTED_OP, small_batch_with_qmin_dynamic_prepacked) {
  FullyConnectedSparseOperatorTester()
      .batchSize(12)
      .inputChannels(23)
      .outputChannels(19)
      .qmin(128)
      .iterations(3)
      .testQ8_prepacked(FullyConnectedSparseOperatorTester::Mode::Dynamic);
}

TEST(FULLY_CONNECTED_OP, small_batch_with_qmax_dynamic_prepacked) {
  FullyConnectedSparseOperatorTester()
      .batchSize(13)
      .inputChannels(23)
      .outputChannels(19)
      .qmax(128)
      .iterations(3)
      .testQ8_prepacked(FullyConnectedSparseOperatorTester::Mode::Dynamic);
}
