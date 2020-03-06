/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "fully-connected-operator-tester.h"

TEST(FULLY_CONNECTED_OP, integration_test_static) {
  FullyConnectedOperatorTester()
      .batchSize(4)
      .inputChannels(4)
      .outputChannels(4)
      .iterations(3)
      .testQ8(FullyConnectedOperatorTester::Mode::Static);
}

TEST(FULLY_CONNECTED_OP, integration_test_runtime) {
  FullyConnectedOperatorTester()
      .batchSize(4)
      .inputChannels(4)
      .outputChannels(4)
      .iterations(3)
      .testQ8(FullyConnectedOperatorTester::Mode::Runtime);
}

TEST(FULLY_CONNECTED_OP, integration_test_dynamic) {
  FullyConnectedOperatorTester()
      .batchSize(4)
      .inputChannels(4)
      .outputChannels(4)
      .iterations(3)
      .testQ8(FullyConnectedOperatorTester::Mode::Dynamic);
}

TEST(FULLY_CONNECTED_OP, zero_batch_static) {
  FullyConnectedOperatorTester()
      .batchSize(0)
      .inputChannels(2)
      .outputChannels(2)
      .iterations(1)
      .testQ8(FullyConnectedOperatorTester::Mode::Static);
}

TEST(FULLY_CONNECTED_OP, zero_batch_runtime) {
  FullyConnectedOperatorTester()
      .batchSize(0)
      .inputChannels(2)
      .outputChannels(2)
      .iterations(1)
      .testQ8(FullyConnectedOperatorTester::Mode::Runtime);
}

TEST(FULLY_CONNECTED_OP, zero_batch_dynamic) {
  FullyConnectedOperatorTester()
      .batchSize(0)
      .inputChannels(2)
      .outputChannels(2)
      .iterations(1)
      .testQ8(FullyConnectedOperatorTester::Mode::Dynamic);
}

TEST(FULLY_CONNECTED_OP, unit_batch_static) {
  FullyConnectedOperatorTester()
      .batchSize(1)
      .inputChannels(23)
      .outputChannels(19)
      .iterations(3)
      .testQ8(FullyConnectedOperatorTester::Mode::Static);
}

TEST(FULLY_CONNECTED_OP, unit_batch_runtime) {
  FullyConnectedOperatorTester()
      .batchSize(1)
      .inputChannels(23)
      .outputChannels(19)
      .iterations(3)
      .testQ8(FullyConnectedOperatorTester::Mode::Runtime);
}

TEST(FULLY_CONNECTED_OP, unit_batch_dynamic) {
  FullyConnectedOperatorTester()
      .batchSize(1)
      .inputChannels(23)
      .outputChannels(19)
      .iterations(3)
      .testQ8(FullyConnectedOperatorTester::Mode::Dynamic);
}

TEST(FULLY_CONNECTED_OP, unit_batch_with_qmin_static) {
  FullyConnectedOperatorTester()
      .batchSize(1)
      .inputChannels(23)
      .outputChannels(19)
      .qmin(128)
      .iterations(3)
      .testQ8(FullyConnectedOperatorTester::Mode::Static);
}

TEST(FULLY_CONNECTED_OP, unit_batch_with_qmin_runtime) {
  FullyConnectedOperatorTester()
      .batchSize(1)
      .inputChannels(23)
      .outputChannels(19)
      .qmin(128)
      .iterations(3)
      .testQ8(FullyConnectedOperatorTester::Mode::Runtime);
}

TEST(FULLY_CONNECTED_OP, unit_batch_with_qmin_dynamic) {
  FullyConnectedOperatorTester()
      .batchSize(1)
      .inputChannels(23)
      .outputChannels(19)
      .qmin(128)
      .iterations(3)
      .testQ8(FullyConnectedOperatorTester::Mode::Dynamic);
}

TEST(FULLY_CONNECTED_OP, unit_batch_with_qmax_static) {
  FullyConnectedOperatorTester()
      .batchSize(1)
      .inputChannels(23)
      .outputChannels(19)
      .qmax(128)
      .iterations(3)
      .testQ8(FullyConnectedOperatorTester::Mode::Static);
}

TEST(FULLY_CONNECTED_OP, unit_batch_with_qmax_runtime) {
  FullyConnectedOperatorTester()
      .batchSize(1)
      .inputChannels(23)
      .outputChannels(19)
      .qmax(128)
      .iterations(3)
      .testQ8(FullyConnectedOperatorTester::Mode::Runtime);
}

TEST(FULLY_CONNECTED_OP, unit_batch_with_qmax_dynamic) {
  FullyConnectedOperatorTester()
      .batchSize(1)
      .inputChannels(23)
      .outputChannels(19)
      .qmax(128)
      .iterations(3)
      .testQ8(FullyConnectedOperatorTester::Mode::Dynamic);
}

TEST(FULLY_CONNECTED_OP, unit_batch_with_input_stride_static) {
  FullyConnectedOperatorTester()
      .batchSize(1)
      .inputChannels(23)
      .inputStride(28)
      .outputChannels(19)
      .iterations(3)
      .testQ8(FullyConnectedOperatorTester::Mode::Static);
}

TEST(FULLY_CONNECTED_OP, unit_batch_with_input_stride_runtime) {
  FullyConnectedOperatorTester()
      .batchSize(1)
      .inputChannels(23)
      .inputStride(28)
      .outputChannels(19)
      .iterations(3)
      .testQ8(FullyConnectedOperatorTester::Mode::Runtime);
}

TEST(FULLY_CONNECTED_OP, unit_batch_with_input_stride_dynamic) {
  FullyConnectedOperatorTester()
      .batchSize(1)
      .inputChannels(23)
      .inputStride(28)
      .outputChannels(19)
      .iterations(3)
      .testQ8(FullyConnectedOperatorTester::Mode::Dynamic);
}

TEST(FULLY_CONNECTED_OP, unit_batch_with_output_stride_static) {
  FullyConnectedOperatorTester()
      .batchSize(1)
      .inputChannels(23)
      .outputChannels(19)
      .outputStride(29)
      .iterations(3)
      .testQ8(FullyConnectedOperatorTester::Mode::Static);
}

TEST(FULLY_CONNECTED_OP, unit_batch_with_output_stride_runtime) {
  FullyConnectedOperatorTester()
      .batchSize(1)
      .inputChannels(23)
      .outputChannels(19)
      .outputStride(29)
      .iterations(3)
      .testQ8(FullyConnectedOperatorTester::Mode::Runtime);
}

TEST(FULLY_CONNECTED_OP, unit_batch_with_output_stride_dynamic) {
  FullyConnectedOperatorTester()
      .batchSize(1)
      .inputChannels(23)
      .outputChannels(19)
      .outputStride(29)
      .iterations(3)
      .testQ8(FullyConnectedOperatorTester::Mode::Dynamic);
}

TEST(FULLY_CONNECTED_OP, small_batch_static) {
  FullyConnectedOperatorTester()
      .batchSize(12)
      .inputChannels(23)
      .outputChannels(19)
      .iterations(3)
      .testQ8(FullyConnectedOperatorTester::Mode::Static);
}

TEST(FULLY_CONNECTED_OP, small_batch_runtime) {
  FullyConnectedOperatorTester()
      .batchSize(12)
      .inputChannels(23)
      .outputChannels(19)
      .iterations(3)
      .testQ8(FullyConnectedOperatorTester::Mode::Runtime);
}

TEST(FULLY_CONNECTED_OP, small_batch_dynamic) {
  FullyConnectedOperatorTester()
      .batchSize(12)
      .inputChannels(23)
      .outputChannels(19)
      .iterations(3)
      .testQ8(FullyConnectedOperatorTester::Mode::Dynamic);
}

TEST(FULLY_CONNECTED_OP, small_batch_with_qmin_static) {
  FullyConnectedOperatorTester()
      .batchSize(12)
      .inputChannels(23)
      .outputChannels(19)
      .qmin(128)
      .iterations(3)
      .testQ8(FullyConnectedOperatorTester::Mode::Static);
}

TEST(FULLY_CONNECTED_OP, small_batch_with_qmin_runtime) {
  FullyConnectedOperatorTester()
      .batchSize(12)
      .inputChannels(23)
      .outputChannels(19)
      .qmin(128)
      .iterations(3)
      .testQ8(FullyConnectedOperatorTester::Mode::Runtime);
}

TEST(FULLY_CONNECTED_OP, small_batch_with_qmin_dynamic) {
  FullyConnectedOperatorTester()
      .batchSize(12)
      .inputChannels(23)
      .outputChannels(19)
      .qmin(128)
      .iterations(3)
      .testQ8(FullyConnectedOperatorTester::Mode::Dynamic);
}

TEST(FULLY_CONNECTED_OP, small_batch_with_qmax) {
  FullyConnectedOperatorTester()
      .batchSize(12)
      .inputChannels(23)
      .outputChannels(19)
      .qmax(128)
      .iterations(3)
      .testQ8(FullyConnectedOperatorTester::Mode::Static);
}

TEST(FULLY_CONNECTED_OP, small_batch_with_qmax_runtime) {
  FullyConnectedOperatorTester()
      .batchSize(12)
      .inputChannels(23)
      .outputChannels(19)
      .qmax(128)
      .iterations(3)
      .testQ8(FullyConnectedOperatorTester::Mode::Runtime);
}

TEST(FULLY_CONNECTED_OP, small_batch_with_qmax_dynamic) {
  FullyConnectedOperatorTester()
      .batchSize(12)
      .inputChannels(23)
      .outputChannels(19)
      .qmax(128)
      .iterations(3)
      .testQ8(FullyConnectedOperatorTester::Mode::Dynamic);
}

TEST(FULLY_CONNECTED_OP, small_batch_with_input_stride_static) {
  FullyConnectedOperatorTester()
      .batchSize(12)
      .inputChannels(23)
      .inputStride(28)
      .outputChannels(19)
      .iterations(3)
      .testQ8(FullyConnectedOperatorTester::Mode::Static);
}

// TODO: Broken
// TEST(FULLY_CONNECTED_OP, small_batch_with_input_stride_runtime) {
//   FullyConnectedOperatorTester()
//       .batchSize(12)
//       .inputChannels(23)
//       .inputStride(28)
//       .outputChannels(19)
//       .iterations(3)
//       .testQ8(FullyConnectedOperatorTester::Mode::Runtime);
// }

// TEST(FULLY_CONNECTED_OP, small_batch_with_input_stride_dynamic) {
//   FullyConnectedOperatorTester()
//       .batchSize(12)
//       .inputChannels(23)
//       .inputStride(28)
//       .outputChannels(19)
//       .iterations(3)
//       .testQ8(FullyConnectedOperatorTester::Mode::Dynamic);
// }

TEST(FULLY_CONNECTED_OP, small_batch_with_output_stride_static) {
  FullyConnectedOperatorTester()
      .batchSize(12)
      .inputChannels(23)
      .outputChannels(19)
      .outputStride(29)
      .iterations(3)
      .testQ8(FullyConnectedOperatorTester::Mode::Static);
}

// TEST(FULLY_CONNECTED_OP, small_batch_with_output_stride_runtime) {
//   FullyConnectedOperatorTester()
//       .batchSize(12)
//       .inputChannels(23)
//       .outputChannels(19)
//       .outputStride(29)
//       .iterations(3)
//       .testQ8(FullyConnectedOperatorTester::Mode::Runtime);
// }

// TEST(FULLY_CONNECTED_OP, small_batch_with_output_stride_dynamic) {
//   FullyConnectedOperatorTester()
//       .batchSize(12)
//       .inputChannels(23)
//       .outputChannels(19)
//       .outputStride(29)
//       .iterations(3)
//       .testQ8(FullyConnectedOperatorTester::Mode::Dynamic);
// }
