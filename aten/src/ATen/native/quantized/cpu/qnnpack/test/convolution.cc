/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <qnnpack/params.h>

#include "convolution-operator-tester.h"

TEST(CONVOLUTION_OP, zero_batch) {
  ConvolutionOperatorTester()
      .batchSize(0)
      .inputSize(5, 5)
      .kernelSize(1, 1)
      .groupInputChannels(2)
      .groupOutputChannels(2)
      .iterations(1)
      .testQ8();
}

TEST(CONVOLUTION_OP, 1x1) {
  ConvolutionOperatorTester()
      .inputSize(27, 29)
      .kernelSize(1, 1)
      .groupInputChannels(23)
      .groupOutputChannels(19)
      .iterations(3)
      .testQ8();
}

TEST(CONVOLUTION_OP, 1x1_runtime_quant) {
  ConvolutionOperatorTester()
      .inputSize(27, 29)
      .kernelSize(1, 1)
      .groupInputChannels(23)
      .groupOutputChannels(19)
      .iterations(3)
      .testQ8(ConvolutionOperatorTester::Mode::Runtime);
}

TEST(CONVOLUTION_OP, 1x1_with_qmin) {
  ConvolutionOperatorTester()
      .inputSize(27, 29)
      .kernelSize(1, 1)
      .groupInputChannels(23)
      .groupOutputChannels(19)
      .qmin(128)
      .iterations(3)
      .testQ8();
}

TEST(CONVOLUTION_OP, 1x1_with_qmax) {
  ConvolutionOperatorTester()
      .inputSize(27, 29)
      .kernelSize(1, 1)
      .groupInputChannels(23)
      .groupOutputChannels(19)
      .qmax(128)
      .iterations(3)
      .testQ8();
}

TEST(CONVOLUTION_OP, 1x1_with_input_stride) {
  ConvolutionOperatorTester()
      .inputSize(27, 29)
      .kernelSize(1, 1)
      .inputPixelStride(28)
      .groupInputChannels(23)
      .groupOutputChannels(19)
      .iterations(3)
      .testQ8();
}

TEST(CONVOLUTION_OP, 1x1_with_output_stride) {
  ConvolutionOperatorTester()
      .inputSize(27, 29)
      .kernelSize(1, 1)
      .outputPixelStride(29)
      .groupInputChannels(23)
      .groupOutputChannels(19)
      .iterations(3)
      .testQ8();
}

TEST(CONVOLUTION_OP, 1x1_with_batch) {
  ConvolutionOperatorTester()
      .inputSize(13, 14)
      .kernelSize(1, 1)
      .batchSize(3)
      .groupInputChannels(23)
      .groupOutputChannels(19)
      .iterations(3)
      .testQ8();
}

TEST(CONVOLUTION_OP, grouped_1x1) {
  ConvolutionOperatorTester()
      .inputSize(24, 25)
      .kernelSize(1, 1)
      .groups(2)
      .groupInputChannels(17)
      .groupOutputChannels(19)
      .iterations(3)
      .testQ8();
}

TEST(CONVOLUTION_OP, xzp_1x1) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  if (pytorch_qnnp_params.q8conv_xzp.kthreshold != SIZE_MAX) {
    ConvolutionOperatorTester()
        .inputSize(27, 29)
        .kernelSize(1, 1)
        .groupInputChannels(pytorch_qnnp_params.q8conv_xzp.kthreshold + 1)
        .groupOutputChannels(19)
        .iterations(3)
        .testQ8();
  }
}

TEST(CONVOLUTION_OP, xzp_1x1_with_qmin) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  if (pytorch_qnnp_params.q8conv_xzp.kthreshold != SIZE_MAX) {
    ConvolutionOperatorTester()
        .inputSize(27, 29)
        .kernelSize(1, 1)
        .groupInputChannels(pytorch_qnnp_params.q8conv_xzp.kthreshold + 1)
        .groupOutputChannels(19)
        .qmin(128)
        .iterations(3)
        .testQ8();
  }
}

TEST(CONVOLUTION_OP, xzp_1x1_with_qmax) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  if (pytorch_qnnp_params.q8conv_xzp.kthreshold != SIZE_MAX) {
    ConvolutionOperatorTester()
        .inputSize(27, 29)
        .kernelSize(1, 1)
        .groupInputChannels(pytorch_qnnp_params.q8conv_xzp.kthreshold + 1)
        .groupOutputChannels(19)
        .qmax(128)
        .iterations(3)
        .testQ8();
  }
}

TEST(CONVOLUTION_OP, xzp_1x1_with_input_stride) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  if (pytorch_qnnp_params.q8conv_xzp.kthreshold != SIZE_MAX) {
    ConvolutionOperatorTester()
        .inputSize(27, 29)
        .kernelSize(1, 1)
        .inputPixelStride(pytorch_qnnp_params.q8conv_xzp.kthreshold + 5)
        .groupInputChannels(pytorch_qnnp_params.q8conv_xzp.kthreshold + 1)
        .groupOutputChannels(19)
        .iterations(3)
        .testQ8();
  }
}

TEST(CONVOLUTION_OP, xzp_1x1_with_output_stride) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  if (pytorch_qnnp_params.q8conv_xzp.kthreshold != SIZE_MAX) {
    ConvolutionOperatorTester()
        .inputSize(27, 29)
        .kernelSize(1, 1)
        .outputPixelStride(29)
        .groupInputChannels(pytorch_qnnp_params.q8conv_xzp.kthreshold + 1)
        .groupOutputChannels(19)
        .iterations(3)
        .testQ8();
  }
}

TEST(CONVOLUTION_OP, xzp_1x1_with_batch) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  if (pytorch_qnnp_params.q8conv_xzp.kthreshold != SIZE_MAX) {
    ConvolutionOperatorTester()
        .inputSize(13, 14)
        .kernelSize(1, 1)
        .batchSize(3)
        .groupInputChannels(pytorch_qnnp_params.q8conv_xzp.kthreshold + 1)
        .groupOutputChannels(19)
        .iterations(3)
        .testQ8();
  }
}

TEST(CONVOLUTION_OP, grouped_xzp_1x1) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  if (pytorch_qnnp_params.q8conv_xzp.kthreshold != SIZE_MAX) {
    ConvolutionOperatorTester()
        .inputSize(24, 25)
        .kernelSize(1, 1)
        .groups(2)
        .groupInputChannels(pytorch_qnnp_params.q8conv_xzp.kthreshold + 1)
        .groupOutputChannels(19)
        .iterations(3)
        .testQ8();
  }
}

TEST(CONVOLUTION_OP, grouped_xzp_1x1_runtime_quant) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  if (pytorch_qnnp_params.q8conv_xzp.kthreshold != SIZE_MAX) {
    ConvolutionOperatorTester()
        .inputSize(24, 25)
        .kernelSize(1, 1)
        .groups(2)
        .groupInputChannels(pytorch_qnnp_params.q8conv_xzp.kthreshold + 1)
        .groupOutputChannels(19)
        .iterations(3)
        .testQ8(ConvolutionOperatorTester::Mode::Runtime);
  }
}

TEST(CONVOLUTION_OP, 1x3) {
  ConvolutionOperatorTester()
      .inputSize(20, 19)
      .paddingWidth(1)
      .kernelSize(1, 3)
      .groupInputChannels(17)
      .groupOutputChannels(15)
      .iterations(3)
      .testQ8();
}

TEST(CONVOLUTION_OP, grouped_1x3) {
  ConvolutionOperatorTester()
      .inputSize(20, 19)
      .paddingWidth(1)
      .kernelSize(1, 3)
      .groups(2)
      .groupInputChannels(17)
      .groupOutputChannels(15)
      .iterations(3)
      .testQ8();
}

TEST(CONVOLUTION_OP, grouped_1x3_runtime_quant) {
  ConvolutionOperatorTester()
      .inputSize(20, 19)
      .paddingWidth(1)
      .kernelSize(1, 3)
      .groups(2)
      .groupInputChannels(17)
      .groupOutputChannels(15)
      .iterations(3)
      .testQ8(ConvolutionOperatorTester::Mode::Runtime);
}

TEST(CONVOLUTION_OP, 3x1) {
  ConvolutionOperatorTester()
      .inputSize(19, 20)
      .paddingHeight(1)
      .kernelSize(3, 1)
      .groupInputChannels(17)
      .groupOutputChannels(15)
      .iterations(3)
      .testQ8();
}

TEST(CONVOLUTION_OP, grouped_3x1) {
  ConvolutionOperatorTester()
      .inputSize(19, 20)
      .paddingHeight(1)
      .kernelSize(3, 1)
      .groups(2)
      .groupInputChannels(17)
      .groupOutputChannels(15)
      .iterations(3)
      .testQ8();
}

TEST(CONVOLUTION_OP, 3x3) {
  ConvolutionOperatorTester()
      .inputSize(13, 12)
      .padding(1)
      .kernelSize(3, 3)
      .groupInputChannels(15)
      .groupOutputChannels(17)
      .iterations(3)
      .testQ8();
}

TEST(CONVOLUTION_OP, 3x3_without_padding) {
  ConvolutionOperatorTester()
      .inputSize(13, 12)
      .kernelSize(3, 3)
      .groupInputChannels(15)
      .groupOutputChannels(17)
      .iterations(3)
      .testQ8();
}

TEST(CONVOLUTION_OP, 3x3_with_left_padding) {
  ConvolutionOperatorTester()
      .inputSize(13, 12)
      .paddingLeft(1)
      .kernelSize(3, 3)
      .groupInputChannels(15)
      .groupOutputChannels(17)
      .iterations(3)
      .testQ8();
}

TEST(CONVOLUTION_OP, 3x3_with_right_padding) {
  ConvolutionOperatorTester()
      .inputSize(13, 12)
      .paddingRight(1)
      .kernelSize(3, 3)
      .groupInputChannels(15)
      .groupOutputChannels(17)
      .iterations(3)
      .testQ8();
}

TEST(CONVOLUTION_OP, 3x3_with_top_padding) {
  ConvolutionOperatorTester()
      .inputSize(13, 12)
      .paddingTop(1)
      .kernelSize(3, 3)
      .groupInputChannels(15)
      .groupOutputChannels(17)
      .iterations(3)
      .testQ8();
}

TEST(CONVOLUTION_OP, 3x3_with_bottom_padding) {
  ConvolutionOperatorTester()
      .inputSize(13, 12)
      .paddingBottom(1)
      .kernelSize(3, 3)
      .groupInputChannels(15)
      .groupOutputChannels(17)
      .iterations(3)
      .testQ8();
}

TEST(CONVOLUTION_OP, 3x3_with_input_stride) {
  ConvolutionOperatorTester()
      .inputSize(13, 12)
      .padding(1)
      .kernelSize(3, 3)
      .inputPixelStride(22)
      .groupInputChannels(15)
      .groupOutputChannels(17)
      .iterations(3)
      .testQ8();
}

TEST(CONVOLUTION_OP, 3x3_with_output_stride) {
  ConvolutionOperatorTester()
      .inputSize(13, 12)
      .padding(1)
      .kernelSize(3, 3)
      .outputPixelStride(23)
      .groupInputChannels(15)
      .groupOutputChannels(17)
      .iterations(3)
      .testQ8();
}

TEST(CONVOLUTION_OP, 3x3_with_batch) {
  ConvolutionOperatorTester()
      .inputSize(10, 9)
      .padding(1)
      .kernelSize(3, 3)
      .batchSize(3)
      .groupInputChannels(15)
      .groupOutputChannels(17)
      .iterations(3)
      .testQ8();
}

TEST(CONVOLUTION_OP, grouped_3x3) {
  ConvolutionOperatorTester()
      .inputSize(10, 11)
      .padding(1)
      .kernelSize(3, 3)
      .groups(2)
      .groupInputChannels(14)
      .groupOutputChannels(13)
      .iterations(3)
      .testQ8();
}

TEST(CONVOLUTION_OP, 3x3s2) {
  ConvolutionOperatorTester()
      .inputSize(19, 21)
      .padding(1)
      .kernelSize(3, 3)
      .subsampling(2)
      .groupInputChannels(27)
      .groupOutputChannels(19)
      .iterations(3)
      .testQ8();
}

TEST(CONVOLUTION_OP, 3x3s1x2) {
  ConvolutionOperatorTester()
      .inputSize(13, 13)
      .padding(1)
      .kernelSize(3, 3)
      .subsampling(1, 2)
      .groupInputChannels(27)
      .groupOutputChannels(19)
      .iterations(3)
      .testQ8();
}

TEST(CONVOLUTION_OP, 3x3s2x1) {
  ConvolutionOperatorTester()
      .inputSize(13, 13)
      .padding(1)
      .kernelSize(3, 3)
      .subsampling(2, 1)
      .groupInputChannels(27)
      .groupOutputChannels(19)
      .iterations(3)
      .testQ8();
}

TEST(CONVOLUTION_OP, 3x3d2) {
  ConvolutionOperatorTester()
      .inputSize(13, 14)
      .padding(2)
      .kernelSize(3, 3)
      .dilation(2)
      .groupInputChannels(27)
      .groupOutputChannels(19)
      .iterations(3)
      .testQ8();
}

TEST(CONVOLUTION_OP, 3x3d1x2) {
  ConvolutionOperatorTester()
      .inputSize(14, 15)
      .padding(1, 2)
      .kernelSize(3, 3)
      .dilation(1, 2)
      .groupInputChannels(27)
      .groupOutputChannels(19)
      .iterations(3)
      .testQ8();
}

TEST(CONVOLUTION_OP, 3x3d2x1) {
  ConvolutionOperatorTester()
      .inputSize(15, 14)
      .padding(2, 1)
      .kernelSize(3, 3)
      .dilation(2, 1)
      .groupInputChannels(27)
      .groupOutputChannels(19)
      .iterations(3)
      .testQ8();
}

TEST(CONVOLUTION_OP, depthwise_3x3) {
  ConvolutionOperatorTester()
      .inputSize(15, 14)
      .padding(1, 1)
      .kernelSize(3, 3)
      .groups(27)
      .iterations(3)
      .testQ8();
}

TEST(CONVOLUTION_OP, depthwise_3x3_runtime_quant) {
  ConvolutionOperatorTester()
      .inputSize(15, 14)
      .padding(1, 1)
      .kernelSize(3, 3)
      .groups(27)
      .iterations(3)
      .testQ8(ConvolutionOperatorTester::Mode::Runtime);
}

TEST(CONVOLUTION_OP, depthwise_3x3s2) {
  ConvolutionOperatorTester()
      .inputSize(15, 14)
      .padding(1, 1)
      .kernelSize(3, 3)
      .subsampling(2)
      .groups(27)
      .iterations(3)
      .testQ8();
}

TEST(CONVOLUTION_OP, depthwise_3x3s1x2) {
  ConvolutionOperatorTester()
      .inputSize(15, 14)
      .padding(1, 1)
      .kernelSize(3, 3)
      .subsampling(1, 2)
      .groups(27)
      .iterations(3)
      .testQ8();
}

TEST(CONVOLUTION_OP, depthwise_3x3s2x1) {
  ConvolutionOperatorTester()
      .inputSize(15, 14)
      .padding(1, 1)
      .kernelSize(3, 3)
      .subsampling(2, 1)
      .groups(27)
      .iterations(3)
      .testQ8();
}

TEST(CONVOLUTION_OP, depthwise_3x3d2) {
  ConvolutionOperatorTester()
      .inputSize(15, 14)
      .padding(1, 1)
      .kernelSize(3, 3)
      .dilation(2)
      .groups(27)
      .iterations(3)
      .testQ8();
}

TEST(CONVOLUTION_OP, depthwise_3x3d1x2) {
  ConvolutionOperatorTester()
      .inputSize(15, 14)
      .padding(1, 1)
      .kernelSize(3, 3)
      .dilation(1, 2)
      .groups(27)
      .iterations(3)
      .testQ8();
}

TEST(CONVOLUTION_OP, depthwise_3x3d2x1) {
  ConvolutionOperatorTester()
      .inputSize(15, 14)
      .padding(1, 1)
      .kernelSize(3, 3)
      .dilation(2, 1)
      .groups(27)
      .iterations(3)
      .testQ8();
}

TEST(CONVOLUTION_OP, depthwise_3x3d2x1_runtime_quant) {
  ConvolutionOperatorTester()
      .inputSize(15, 14)
      .padding(1, 1)
      .kernelSize(3, 3)
      .dilation(2, 1)
      .groups(27)
      .iterations(3)
      .testQ8(ConvolutionOperatorTester::Mode::Runtime);
}

TEST(CONVOLUTION_OP, depthwise_5x5) {
  ConvolutionOperatorTester()
      .inputSize(15, 14)
      .padding(2, 2)
      .kernelSize(5, 5)
      .groups(27)
      .iterations(3)
      .testQ8();
}

TEST(CONVOLUTION_OP, depthwise_5x5s2) {
  ConvolutionOperatorTester()
      .inputSize(15, 14)
      .padding(2, 2)
      .kernelSize(5, 5)
      .subsampling(2)
      .groups(27)
      .iterations(3)
      .testQ8();
}

TEST(CONVOLUTION_OP, depthwise_5x5s1x2) {
  ConvolutionOperatorTester()
      .inputSize(15, 14)
      .padding(2, 2)
      .kernelSize(5, 5)
      .subsampling(1, 2)
      .groups(27)
      .iterations(3)
      .testQ8();
}

TEST(CONVOLUTION_OP, depthwise_5x5s2x1) {
  ConvolutionOperatorTester()
      .inputSize(15, 14)
      .padding(2, 2)
      .kernelSize(5, 5)
      .subsampling(2, 1)
      .groups(27)
      .iterations(3)
      .testQ8();
}

TEST(CONVOLUTION_OP, depthwise_5x5d2) {
  ConvolutionOperatorTester()
      .inputSize(15, 14)
      .padding(2, 2)
      .kernelSize(5, 5)
      .dilation(2)
      .groups(27)
      .iterations(3)
      .testQ8();
}

TEST(CONVOLUTION_OP, depthwise_5x5d1x2) {
  ConvolutionOperatorTester()
      .inputSize(15, 14)
      .padding(2, 2)
      .kernelSize(5, 5)
      .dilation(1, 2)
      .groups(27)
      .iterations(3)
      .testQ8();
}

TEST(CONVOLUTION_OP, depthwise_5x5d2x1) {
  ConvolutionOperatorTester()
      .inputSize(15, 14)
      .padding(2, 2)
      .kernelSize(5, 5)
      .dilation(2, 1)
      .groups(27)
      .iterations(3)
      .testQ8();
}

TEST(CONVOLUTION_OP, depthwise_5x5d2x1_runtime_quant) {
  ConvolutionOperatorTester()
      .inputSize(15, 14)
      .padding(2, 2)
      .kernelSize(5, 5)
      .dilation(2, 1)
      .groups(27)
      .iterations(3)
      .testQ8(ConvolutionOperatorTester::Mode::Runtime);
}
