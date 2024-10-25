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

using namespace qnnpack::testing;

_STATIC_AND_RUNTIME_TEST(CONVOLUTION_OP, zero_batch,
  ConvolutionOperatorTester()
      .batchSize(0)
      .inputSize(5, 5)
      .kernelSize(1, 1)
      .groupInputChannels(2)
      .groupOutputChannels(2)
      .iterations(1)
)

_STATIC_AND_RUNTIME_TEST(CONVOLUTION_OP, 1x1,
  ConvolutionOperatorTester()
      .inputSize(27, 29)
      .kernelSize(1, 1)
      .groupInputChannels(23)
      .groupOutputChannels(19)
      .iterations(3)
)

_STATIC_AND_RUNTIME_TEST(CONVOLUTION_OP, 1x1_runtime_quant,
  ConvolutionOperatorTester()
      .inputSize(27, 29)
      .kernelSize(1, 1)
      .groupInputChannels(23)
      .groupOutputChannels(19)
      .iterations(3)
)

_STATIC_AND_RUNTIME_TEST(CONVOLUTION_OP, 1x1_with_qmin,
  ConvolutionOperatorTester()
      .inputSize(27, 29)
      .kernelSize(1, 1)
      .groupInputChannels(23)
      .groupOutputChannels(19)
      .qmin(128)
      .iterations(3)
)

_STATIC_AND_RUNTIME_TEST(CONVOLUTION_OP, 1x1_with_qmax,
  ConvolutionOperatorTester()
      .inputSize(27, 29)
      .kernelSize(1, 1)
      .groupInputChannels(23)
      .groupOutputChannels(19)
      .qmax(128)
      .iterations(3)
)

_STATIC_TEST(CONVOLUTION_OP, 1x1_with_input_stride,
  ConvolutionOperatorTester()
      .inputSize(27, 29)
      .kernelSize(1, 1)
      .inputPixelStride(28)
      .groupInputChannels(23)
      .groupOutputChannels(19)
      .iterations(3)
)

_STATIC_TEST(CONVOLUTION_OP, 1x1_with_output_stride,
  ConvolutionOperatorTester()
      .inputSize(27, 29)
      .kernelSize(1, 1)
      .outputPixelStride(29)
      .groupInputChannels(23)
      .groupOutputChannels(19)
      .iterations(3)
)

_STATIC_AND_RUNTIME_TEST(CONVOLUTION_OP, 1x1_with_batch,
  ConvolutionOperatorTester()
      .inputSize(13, 14)
      .kernelSize(1, 1)
      .batchSize(3)
      .groupInputChannels(23)
      .groupOutputChannels(19)
      .iterations(3)
)

_STATIC_AND_RUNTIME_TEST(CONVOLUTION_OP, grouped_1x1,
  ConvolutionOperatorTester()
      .inputSize(24, 25)
      .kernelSize(1, 1)
      .groups(2)
      .groupInputChannels(17)
      .groupOutputChannels(19)
      .iterations(3)
)

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
        .testQ8(Mode::Runtime);
  }
}

_STATIC_AND_RUNTIME_TEST(CONVOLUTION_OP, 1x3,
  ConvolutionOperatorTester()
      .inputSize(20, 19)
      .paddingWidth(1)
      .kernelSize(1, 3)
      .groupInputChannels(17)
      .groupOutputChannels(15)
      .iterations(3)
)

_STATIC_AND_RUNTIME_TEST(CONVOLUTION_OP, grouped_1x3,
  ConvolutionOperatorTester()
      .inputSize(20, 19)
      .paddingWidth(1)
      .kernelSize(1, 3)
      .groups(2)
      .groupInputChannels(17)
      .groupOutputChannels(15)
      .iterations(3)
)

_STATIC_AND_RUNTIME_TEST(CONVOLUTION_OP, 3x1,
  ConvolutionOperatorTester()
      .inputSize(19, 20)
      .paddingHeight(1)
      .kernelSize(3, 1)
      .groupInputChannels(17)
      .groupOutputChannels(15)
      .iterations(3)
)

_STATIC_AND_RUNTIME_TEST(CONVOLUTION_OP, grouped_3x1,
  ConvolutionOperatorTester()
      .inputSize(19, 20)
      .paddingHeight(1)
      .kernelSize(3, 1)
      .groups(2)
      .groupInputChannels(17)
      .groupOutputChannels(15)
      .iterations(3)
)

_STATIC_AND_RUNTIME_TEST(CONVOLUTION_OP, 3x3,
  ConvolutionOperatorTester()
      .inputSize(13, 12)
      .padding(1)
      .kernelSize(3, 3)
      .groupInputChannels(15)
      .groupOutputChannels(17)
      .iterations(3)
)

_STATIC_AND_RUNTIME_TEST(CONVOLUTION_OP, 3x3_without_padding,
  ConvolutionOperatorTester()
      .inputSize(13, 12)
      .kernelSize(3, 3)
      .groupInputChannels(15)
      .groupOutputChannels(17)
      .iterations(3)
)

_STATIC_AND_RUNTIME_TEST(
    CONVOLUTION_OP,
    3x3_with_width_padding,
    ConvolutionOperatorTester()
        .inputSize(13, 12)
        .paddingWidth(1)
        .kernelSize(3, 3)
        .groupInputChannels(15)
        .groupOutputChannels(17)
        .iterations(3))

_STATIC_AND_RUNTIME_TEST(
    CONVOLUTION_OP,
    3x3_with_height_padding,
    ConvolutionOperatorTester()
        .inputSize(13, 12)
        .paddingHeight(1)
        .kernelSize(3, 3)
        .groupInputChannels(15)
        .groupOutputChannels(17)
        .iterations(3))

_STATIC_TEST(CONVOLUTION_OP, 3x3_with_input_stride,
  ConvolutionOperatorTester()
      .inputSize(13, 12)
      .padding(1)
      .kernelSize(3, 3)
      .inputPixelStride(22)
      .groupInputChannels(15)
      .groupOutputChannels(17)
      .iterations(3)
)

_STATIC_TEST(CONVOLUTION_OP, 3x3_with_output_stride,
  ConvolutionOperatorTester()
      .inputSize(13, 12)
      .padding(1)
      .kernelSize(3, 3)
      .outputPixelStride(23)
      .groupInputChannels(15)
      .groupOutputChannels(17)
      .iterations(3)
)

_STATIC_AND_RUNTIME_TEST(CONVOLUTION_OP, 3x3_with_batch,
  ConvolutionOperatorTester()
      .inputSize(10, 9)
      .padding(1)
      .kernelSize(3, 3)
      .batchSize(3)
      .groupInputChannels(15)
      .groupOutputChannels(17)
      .iterations(3)
)

_STATIC_AND_RUNTIME_TEST(CONVOLUTION_OP, grouped_3x3,
  ConvolutionOperatorTester()
      .inputSize(10, 11)
      .padding(1)
      .kernelSize(3, 3)
      .groups(2)
      .groupInputChannels(14)
      .groupOutputChannels(13)
      .iterations(3)
)

_STATIC_AND_RUNTIME_TEST(CONVOLUTION_OP, 3x3s2,
  ConvolutionOperatorTester()
      .inputSize(19, 21)
      .padding(1)
      .kernelSize(3, 3)
      .subsampling(2)
      .groupInputChannels(27)
      .groupOutputChannels(19)
      .iterations(3)
)

_STATIC_AND_RUNTIME_TEST(CONVOLUTION_OP, 3x3s1x2,
  ConvolutionOperatorTester()
      .inputSize(13, 13)
      .padding(1)
      .kernelSize(3, 3)
      .subsampling(1, 2)
      .groupInputChannels(27)
      .groupOutputChannels(19)
      .iterations(3)
)

_STATIC_AND_RUNTIME_TEST(CONVOLUTION_OP, 3x3s2x1,
  ConvolutionOperatorTester()
      .inputSize(13, 13)
      .padding(1)
      .kernelSize(3, 3)
      .subsampling(2, 1)
      .groupInputChannels(27)
      .groupOutputChannels(19)
      .iterations(3)
)

_STATIC_AND_RUNTIME_TEST(CONVOLUTION_OP, 3x3d2,
  ConvolutionOperatorTester()
      .inputSize(13, 14)
      .padding(2)
      .kernelSize(3, 3)
      .dilation(2)
      .groupInputChannels(27)
      .groupOutputChannels(19)
      .iterations(3)
)

_STATIC_AND_RUNTIME_TEST(CONVOLUTION_OP, 3x3d1x2,
  ConvolutionOperatorTester()
      .inputSize(14, 15)
      .padding(1, 2)
      .kernelSize(3, 3)
      .dilation(1, 2)
      .groupInputChannels(27)
      .groupOutputChannels(19)
      .iterations(3)
)

_STATIC_AND_RUNTIME_TEST(CONVOLUTION_OP, 3x3d2x1,
  ConvolutionOperatorTester()
      .inputSize(15, 14)
      .padding(2, 1)
      .kernelSize(3, 3)
      .dilation(2, 1)
      .groupInputChannels(27)
      .groupOutputChannels(19)
      .iterations(3)
)

_STATIC_AND_RUNTIME_TEST(CONVOLUTION_OP, depthwise_3x3,
  ConvolutionOperatorTester()
      .inputSize(15, 14)
      .padding(1, 1)
      .kernelSize(3, 3)
      .groups(27)
      .iterations(3)
)

_STATIC_AND_RUNTIME_TEST(CONVOLUTION_OP, depthwise_3x3s2,
  ConvolutionOperatorTester()
      .inputSize(15, 14)
      .padding(1, 1)
      .kernelSize(3, 3)
      .subsampling(2)
      .groups(27)
      .iterations(3)
)

_STATIC_AND_RUNTIME_TEST(CONVOLUTION_OP, depthwise_3x3s1x2,
  ConvolutionOperatorTester()
      .inputSize(15, 14)
      .padding(1, 1)
      .kernelSize(3, 3)
      .subsampling(1, 2)
      .groups(27)
      .iterations(3)
)

_STATIC_AND_RUNTIME_TEST(CONVOLUTION_OP, depthwise_3x3s2x1,
  ConvolutionOperatorTester()
      .inputSize(15, 14)
      .padding(1, 1)
      .kernelSize(3, 3)
      .subsampling(2, 1)
      .groups(27)
      .iterations(3)
)

_STATIC_AND_RUNTIME_TEST(CONVOLUTION_OP, depthwise_3x3d2,
  ConvolutionOperatorTester()
      .inputSize(15, 14)
      .padding(1, 1)
      .kernelSize(3, 3)
      .dilation(2)
      .groups(27)
      .iterations(3)
)

_STATIC_AND_RUNTIME_TEST(CONVOLUTION_OP, depthwise_3x3d1x2,
  ConvolutionOperatorTester()
      .inputSize(15, 14)
      .padding(1, 1)
      .kernelSize(3, 3)
      .dilation(1, 2)
      .groups(27)
      .iterations(3)
)

_STATIC_AND_RUNTIME_TEST(CONVOLUTION_OP, depthwise_3x3d2x1,
  ConvolutionOperatorTester()
      .inputSize(15, 14)
      .padding(1, 1)
      .kernelSize(3, 3)
      .dilation(2, 1)
      .groups(27)
      .iterations(3)
)

_STATIC_AND_RUNTIME_TEST(CONVOLUTION_OP, depthwise_5x5,
  ConvolutionOperatorTester()
      .inputSize(15, 14)
      .padding(2, 2)
      .kernelSize(5, 5)
      .groups(27)
      .iterations(3)
)

_STATIC_AND_RUNTIME_TEST(CONVOLUTION_OP, depthwise_5x5s2,
  ConvolutionOperatorTester()
      .inputSize(15, 14)
      .padding(2, 2)
      .kernelSize(5, 5)
      .subsampling(2)
      .groups(27)
      .iterations(3)
)

_STATIC_AND_RUNTIME_TEST(CONVOLUTION_OP, depthwise_5x5s1x2,
  ConvolutionOperatorTester()
      .inputSize(15, 14)
      .padding(2, 2)
      .kernelSize(5, 5)
      .subsampling(1, 2)
      .groups(27)
      .iterations(3)
)

_STATIC_AND_RUNTIME_TEST(CONVOLUTION_OP, depthwise_5x5s2x1,
  ConvolutionOperatorTester()
      .inputSize(15, 14)
      .padding(2, 2)
      .kernelSize(5, 5)
      .subsampling(2, 1)
      .groups(27)
      .iterations(3)
)

_STATIC_AND_RUNTIME_TEST(CONVOLUTION_OP, depthwise_5x5d2,
  ConvolutionOperatorTester()
      .inputSize(15, 14)
      .padding(2, 2)
      .kernelSize(5, 5)
      .dilation(2)
      .groups(27)
      .iterations(3)
)

_STATIC_AND_RUNTIME_TEST(CONVOLUTION_OP, depthwise_5x5d1x2,
  ConvolutionOperatorTester()
      .inputSize(15, 14)
      .padding(2, 2)
      .kernelSize(5, 5)
      .dilation(1, 2)
      .groups(27)
      .iterations(3)
)

_STATIC_AND_RUNTIME_TEST(CONVOLUTION_OP, depthwise_5x5d2x1,
  ConvolutionOperatorTester()
      .inputSize(15, 14)
      .padding(2, 2)
      .kernelSize(5, 5)
      .dilation(2, 1)
      .groups(27)
      .iterations(3)
)

_STATIC_AND_RUNTIME_TEST(CONVOLUTION_OP, zero_batch_per_channel,
  ConvolutionOperatorTester()
      .batchSize(0)
      .inputSize(5, 5)
      .kernelSize(1, 1)
      .groupInputChannels(2)
      .groupOutputChannels(2)
      .iterations(1)
      .per_channel(true)
)

_STATIC_AND_RUNTIME_TEST(CONVOLUTION_OP, 1x1_per_channel,
  ConvolutionOperatorTester()
      .inputSize(27, 29)
      .kernelSize(1, 1)
      .groupInputChannels(23)
      .groupOutputChannels(19)
      .iterations(3)
      .per_channel(true)
)

_STATIC_AND_RUNTIME_TEST(CONVOLUTION_OP, 1x1_with_qmin_per_channel,
  ConvolutionOperatorTester()
      .inputSize(27, 29)
      .kernelSize(1, 1)
      .groupInputChannels(23)
      .groupOutputChannels(19)
      .qmin(128)
      .iterations(3)
      .per_channel(true)
)

_STATIC_AND_RUNTIME_TEST(CONVOLUTION_OP, 1x1_with_qmax_per_channel,
  ConvolutionOperatorTester()
      .inputSize(27, 29)
      .kernelSize(1, 1)
      .groupInputChannels(23)
      .groupOutputChannels(19)
      .qmax(128)
      .iterations(3)
      .per_channel(true)
)

_STATIC_TEST(CONVOLUTION_OP, 1x1_with_input_stride_per_channel,
  ConvolutionOperatorTester()
      .inputSize(27, 29)
      .kernelSize(1, 1)
      .inputPixelStride(28)
      .groupInputChannels(23)
      .groupOutputChannels(19)
      .iterations(3)
      .per_channel(true)
)

_STATIC_TEST(CONVOLUTION_OP, 1x1_with_output_stride_per_channel,
  ConvolutionOperatorTester()
      .inputSize(27, 29)
      .kernelSize(1, 1)
      .outputPixelStride(29)
      .groupInputChannels(23)
      .groupOutputChannels(19)
      .iterations(3)
      .per_channel(true)
)

_STATIC_AND_RUNTIME_TEST(CONVOLUTION_OP, 1x1_with_batch_per_channel,
  ConvolutionOperatorTester()
      .inputSize(13, 14)
      .kernelSize(1, 1)
      .batchSize(3)
      .groupInputChannels(23)
      .groupOutputChannels(19)
      .iterations(3)
      .per_channel(true)
)

_STATIC_AND_RUNTIME_TEST(CONVOLUTION_OP, grouped_1x1_per_channel,
  ConvolutionOperatorTester()
      .inputSize(24, 25)
      .kernelSize(1, 1)
      .groups(2)
      .groupInputChannels(17)
      .groupOutputChannels(19)
      .iterations(3)
      .per_channel(true)
)

TEST(CONVOLUTION_OP, xzp_1x1_per_channel) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  if (pytorch_qnnp_params.q8conv_xzp.kthreshold != SIZE_MAX) {
    ConvolutionOperatorTester()
        .inputSize(27, 29)
        .kernelSize(1, 1)
        .groupInputChannels(pytorch_qnnp_params.q8conv_xzp.kthreshold + 1)
        .groupOutputChannels(19)
        .iterations(3)
        .per_channel(true)
        .testQ8();
  }
}

TEST(CONVOLUTION_OP, xzp_1x1_with_qmin_per_channel) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  if (pytorch_qnnp_params.q8conv_xzp.kthreshold != SIZE_MAX) {
    ConvolutionOperatorTester()
        .inputSize(27, 29)
        .kernelSize(1, 1)
        .groupInputChannels(pytorch_qnnp_params.q8conv_xzp.kthreshold + 1)
        .groupOutputChannels(19)
        .qmin(128)
        .iterations(3)
        .per_channel(true)
        .testQ8();
  }
}

TEST(CONVOLUTION_OP, xzp_1x1_with_qmax_per_channel) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  if (pytorch_qnnp_params.q8conv_xzp.kthreshold != SIZE_MAX) {
    ConvolutionOperatorTester()
        .inputSize(27, 29)
        .kernelSize(1, 1)
        .groupInputChannels(pytorch_qnnp_params.q8conv_xzp.kthreshold + 1)
        .groupOutputChannels(19)
        .qmax(128)
        .iterations(3)
        .per_channel(true)
        .testQ8();
  }
}

TEST(CONVOLUTION_OP, xzp_1x1_with_input_stride_per_channel) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  if (pytorch_qnnp_params.q8conv_xzp.kthreshold != SIZE_MAX) {
    ConvolutionOperatorTester()
        .inputSize(27, 29)
        .kernelSize(1, 1)
        .inputPixelStride(pytorch_qnnp_params.q8conv_xzp.kthreshold + 5)
        .groupInputChannels(pytorch_qnnp_params.q8conv_xzp.kthreshold + 1)
        .groupOutputChannels(19)
        .iterations(3)
        .per_channel(true)
        .testQ8();
  }
}

TEST(CONVOLUTION_OP, xzp_1x1_with_output_stride_per_channel) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  if (pytorch_qnnp_params.q8conv_xzp.kthreshold != SIZE_MAX) {
    ConvolutionOperatorTester()
        .inputSize(27, 29)
        .kernelSize(1, 1)
        .outputPixelStride(29)
        .groupInputChannels(pytorch_qnnp_params.q8conv_xzp.kthreshold + 1)
        .groupOutputChannels(19)
        .iterations(3)
        .per_channel(true)
        .testQ8();
  }
}

TEST(CONVOLUTION_OP, xzp_1x1_with_batch_per_channel) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  if (pytorch_qnnp_params.q8conv_xzp.kthreshold != SIZE_MAX) {
    ConvolutionOperatorTester()
        .inputSize(13, 14)
        .kernelSize(1, 1)
        .batchSize(3)
        .groupInputChannels(pytorch_qnnp_params.q8conv_xzp.kthreshold + 1)
        .groupOutputChannels(19)
        .iterations(3)
        .per_channel(true)
        .testQ8();
  }
}

TEST(CONVOLUTION_OP, grouped_xzp_1x1_per_channel) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  if (pytorch_qnnp_params.q8conv_xzp.kthreshold != SIZE_MAX) {
    ConvolutionOperatorTester()
        .inputSize(24, 25)
        .kernelSize(1, 1)
        .groups(2)
        .groupInputChannels(pytorch_qnnp_params.q8conv_xzp.kthreshold + 1)
        .groupOutputChannels(19)
        .iterations(3)
        .per_channel(true)
        .testQ8();
  }
}

TEST(CONVOLUTION_OP, grouped_xzp_1x1_runtime_quant_per_channel) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  if (pytorch_qnnp_params.q8conv_xzp.kthreshold != SIZE_MAX) {
    ConvolutionOperatorTester()
        .inputSize(24, 25)
        .kernelSize(1, 1)
        .groups(2)
        .groupInputChannels(pytorch_qnnp_params.q8conv_xzp.kthreshold + 1)
        .groupOutputChannels(19)
        .iterations(3)
        .per_channel(true)
        .testQ8(Mode::Runtime);
  }
}

_STATIC_AND_RUNTIME_TEST(CONVOLUTION_OP, 1x3_per_channel,
  ConvolutionOperatorTester()
      .inputSize(20, 19)
      .paddingWidth(1)
      .kernelSize(1, 3)
      .groupInputChannels(17)
      .groupOutputChannels(15)
      .iterations(3)
      .per_channel(true)
)

_STATIC_AND_RUNTIME_TEST(CONVOLUTION_OP, grouped_1x3_per_channel,
  ConvolutionOperatorTester()
      .inputSize(20, 19)
      .paddingWidth(1)
      .kernelSize(1, 3)
      .groups(2)
      .groupInputChannels(17)
      .groupOutputChannels(15)
      .iterations(3)
      .per_channel(true)
)

_STATIC_AND_RUNTIME_TEST(CONVOLUTION_OP, 3x1_per_channel,
  ConvolutionOperatorTester()
      .inputSize(19, 20)
      .paddingHeight(1)
      .kernelSize(3, 1)
      .groupInputChannels(17)
      .groupOutputChannels(15)
      .iterations(3)
      .per_channel(true)
)

_STATIC_AND_RUNTIME_TEST(CONVOLUTION_OP, grouped_3x1_per_channel,
  ConvolutionOperatorTester()
      .inputSize(19, 20)
      .paddingHeight(1)
      .kernelSize(3, 1)
      .groups(2)
      .groupInputChannels(17)
      .groupOutputChannels(15)
      .iterations(3)
      .per_channel(true)
)

_STATIC_AND_RUNTIME_TEST(CONVOLUTION_OP, 3x3_per_channel,
  ConvolutionOperatorTester()
      .inputSize(13, 12)
      .padding(1)
      .kernelSize(3, 3)
      .groupInputChannels(15)
      .groupOutputChannels(17)
      .iterations(3)
      .per_channel(true)
)

_STATIC_AND_RUNTIME_TEST(CONVOLUTION_OP, 3x3_without_padding_per_channel,
  ConvolutionOperatorTester()
      .inputSize(13, 12)
      .kernelSize(3, 3)
      .groupInputChannels(15)
      .groupOutputChannels(17)
      .iterations(3)
      .per_channel(true)
)

_STATIC_AND_RUNTIME_TEST(
    CONVOLUTION_OP,
    3x3_with_width_padding_per_channel,
    ConvolutionOperatorTester()
        .inputSize(13, 12)
        .paddingWidth(1)
        .kernelSize(3, 3)
        .groupInputChannels(15)
        .groupOutputChannels(17)
        .iterations(3)
        .per_channel(true))

_STATIC_AND_RUNTIME_TEST(
    CONVOLUTION_OP,
    3x3_with_height_padding_per_channel,
    ConvolutionOperatorTester()
        .inputSize(13, 12)
        .paddingHeight(1)
        .kernelSize(3, 3)
        .groupInputChannels(15)
        .groupOutputChannels(17)
        .iterations(3)
        .per_channel(true))

_STATIC_TEST(CONVOLUTION_OP, 3x3_with_input_stride_per_channel,
  ConvolutionOperatorTester()
      .inputSize(13, 12)
      .padding(1)
      .kernelSize(3, 3)
      .inputPixelStride(22)
      .groupInputChannels(15)
      .groupOutputChannels(17)
      .iterations(3)
      .per_channel(true)
)

_STATIC_TEST(CONVOLUTION_OP, 3x3_with_output_stride_per_channel,
  ConvolutionOperatorTester()
      .inputSize(13, 12)
      .padding(1)
      .kernelSize(3, 3)
      .outputPixelStride(23)
      .groupInputChannels(15)
      .groupOutputChannels(17)
      .iterations(3)
      .per_channel(true)
)

_STATIC_AND_RUNTIME_TEST(CONVOLUTION_OP, 3x3_with_batch_per_channel,
  ConvolutionOperatorTester()
      .inputSize(10, 9)
      .padding(1)
      .kernelSize(3, 3)
      .batchSize(3)
      .groupInputChannels(15)
      .groupOutputChannels(17)
      .iterations(3)
      .per_channel(true)
)

_STATIC_AND_RUNTIME_TEST(CONVOLUTION_OP, grouped_3x3_per_channel,
  ConvolutionOperatorTester()
      .inputSize(10, 11)
      .padding(1)
      .kernelSize(3, 3)
      .groups(2)
      .groupInputChannels(14)
      .groupOutputChannels(13)
      .iterations(3)
      .per_channel(true)
)

_STATIC_AND_RUNTIME_TEST(CONVOLUTION_OP, 3x3s2_per_channel,
  ConvolutionOperatorTester()
      .inputSize(19, 21)
      .padding(1)
      .kernelSize(3, 3)
      .subsampling(2)
      .groupInputChannels(27)
      .groupOutputChannels(19)
      .iterations(3)
      .per_channel(true)
)

_STATIC_AND_RUNTIME_TEST(CONVOLUTION_OP, 3x3s1x2_per_channel,
  ConvolutionOperatorTester()
      .inputSize(13, 13)
      .padding(1)
      .kernelSize(3, 3)
      .subsampling(1, 2)
      .groupInputChannels(27)
      .groupOutputChannels(19)
      .iterations(3)
      .per_channel(true)
)

_STATIC_AND_RUNTIME_TEST(CONVOLUTION_OP, 3x3s2x1_per_channel,
  ConvolutionOperatorTester()
      .inputSize(13, 13)
      .padding(1)
      .kernelSize(3, 3)
      .subsampling(2, 1)
      .groupInputChannels(27)
      .groupOutputChannels(19)
      .iterations(3)
      .per_channel(true)
)

_STATIC_AND_RUNTIME_TEST(CONVOLUTION_OP, 3x3d2_per_channel,
  ConvolutionOperatorTester()
      .inputSize(13, 14)
      .padding(2)
      .kernelSize(3, 3)
      .dilation(2)
      .groupInputChannels(27)
      .groupOutputChannels(19)
      .iterations(3)
      .per_channel(true)
)

_STATIC_AND_RUNTIME_TEST(CONVOLUTION_OP, 3x3d1x2_per_channel,
  ConvolutionOperatorTester()
      .inputSize(14, 15)
      .padding(1, 2)
      .kernelSize(3, 3)
      .dilation(1, 2)
      .groupInputChannels(27)
      .groupOutputChannels(19)
      .iterations(3)
      .per_channel(true)
)

_STATIC_AND_RUNTIME_TEST(CONVOLUTION_OP, 3x3d2x1_per_channel,
  ConvolutionOperatorTester()
      .inputSize(15, 14)
      .padding(2, 1)
      .kernelSize(3, 3)
      .dilation(2, 1)
      .groupInputChannels(27)
      .groupOutputChannels(19)
      .iterations(3)
      .per_channel(true)
)

_STATIC_AND_RUNTIME_TEST(CONVOLUTION_OP, depthwise_3x3_per_channel,
  ConvolutionOperatorTester()
      .inputSize(15, 14)
      .padding(1, 1)
      .kernelSize(3, 3)
      .groups(27)
      .iterations(3)
      .per_channel(true)
)

_STATIC_AND_RUNTIME_TEST(CONVOLUTION_OP, depthwise_3x3s2_per_channel,
  ConvolutionOperatorTester()
      .inputSize(15, 14)
      .padding(1, 1)
      .kernelSize(3, 3)
      .subsampling(2)
      .groups(27)
      .iterations(3)
      .per_channel(true)
)

_STATIC_AND_RUNTIME_TEST(CONVOLUTION_OP, depthwise_3x3s1x2_per_channel,
  ConvolutionOperatorTester()
      .inputSize(15, 14)
      .padding(1, 1)
      .kernelSize(3, 3)
      .subsampling(1, 2)
      .groups(27)
      .iterations(3)
      .per_channel(true)
)

_STATIC_AND_RUNTIME_TEST(CONVOLUTION_OP, depthwise_3x3s2x1_per_channel,
  ConvolutionOperatorTester()
      .inputSize(15, 14)
      .padding(1, 1)
      .kernelSize(3, 3)
      .subsampling(2, 1)
      .groups(27)
      .iterations(3)
      .per_channel(true)
)

_STATIC_AND_RUNTIME_TEST(CONVOLUTION_OP, depthwise_3x3d2_per_channel,
  ConvolutionOperatorTester()
      .inputSize(15, 14)
      .padding(1, 1)
      .kernelSize(3, 3)
      .dilation(2)
      .groups(27)
      .iterations(3)
      .per_channel(true)
)

_STATIC_AND_RUNTIME_TEST(CONVOLUTION_OP, depthwise_3x3d1x2_per_channel,
  ConvolutionOperatorTester()
      .inputSize(15, 14)
      .padding(1, 1)
      .kernelSize(3, 3)
      .dilation(1, 2)
      .groups(27)
      .iterations(3)
      .per_channel(true)
)

_STATIC_AND_RUNTIME_TEST(CONVOLUTION_OP, depthwise_3x3d2x1_per_channel,
  ConvolutionOperatorTester()
      .inputSize(15, 14)
      .padding(1, 1)
      .kernelSize(3, 3)
      .dilation(2, 1)
      .groups(27)
      .iterations(3)
      .per_channel(true)
)

_STATIC_AND_RUNTIME_TEST(CONVOLUTION_OP, depthwise_5x5_per_channel,
  ConvolutionOperatorTester()
      .inputSize(15, 14)
      .padding(2, 2)
      .kernelSize(5, 5)
      .groups(27)
      .iterations(3)
      .per_channel(true)
)

_STATIC_AND_RUNTIME_TEST(CONVOLUTION_OP, depthwise_5x5s2_per_channel,
  ConvolutionOperatorTester()
      .inputSize(15, 14)
      .padding(2, 2)
      .kernelSize(5, 5)
      .subsampling(2)
      .groups(27)
      .iterations(3)
      .per_channel(true)
)

_STATIC_AND_RUNTIME_TEST(CONVOLUTION_OP, depthwise_5x5s1x2_per_channel,
  ConvolutionOperatorTester()
      .inputSize(15, 14)
      .padding(2, 2)
      .kernelSize(5, 5)
      .subsampling(1, 2)
      .groups(27)
      .iterations(3)
      .per_channel(true)
)

_STATIC_AND_RUNTIME_TEST(CONVOLUTION_OP, depthwise_5x5s2x1_per_channel,
  ConvolutionOperatorTester()
      .inputSize(15, 14)
      .padding(2, 2)
      .kernelSize(5, 5)
      .subsampling(2, 1)
      .groups(27)
      .iterations(3)
      .per_channel(true)
)

_STATIC_AND_RUNTIME_TEST(CONVOLUTION_OP, depthwise_5x5d2_per_channel,
  ConvolutionOperatorTester()
      .inputSize(15, 14)
      .padding(2, 2)
      .kernelSize(5, 5)
      .dilation(2)
      .groups(27)
      .iterations(3)
      .per_channel(true)
)

_STATIC_AND_RUNTIME_TEST(CONVOLUTION_OP, depthwise_5x5d1x2_per_channel,
  ConvolutionOperatorTester()
      .inputSize(15, 14)
      .padding(2, 2)
      .kernelSize(5, 5)
      .dilation(1, 2)
      .groups(27)
      .iterations(3)
)

_STATIC_AND_RUNTIME_TEST(CONVOLUTION_OP, depthwise_5x5d2x1_per_channel,
  ConvolutionOperatorTester()
      .inputSize(15, 14)
      .padding(2, 2)
      .kernelSize(5, 5)
      .dilation(2, 1)
      .groups(27)
      .iterations(3)
      .per_channel(true)
)

_STATIC_AND_RUNTIME_TEST(
    CONVOLUTION_3D_OP,
    zero_batch,
    ConvolutionOperatorTester()
        .dimensionality(3)
        .batchSize(0)
        .inputSize(5, 5, 5)
        .kernelSize(1, 1, 1)
        .groupInputChannels(2)
        .groupOutputChannels(2)
        .iterations(1))

_STATIC_AND_RUNTIME_TEST(
    CONVOLUTION_3D_OP,
    1x1x1,
    ConvolutionOperatorTester()
        .dimensionality(3)
        .inputSize(8, 9, 10)
        .kernelSize(1, 1, 1)
        .groupInputChannels(5)
        .groupInputChannels(6)
        .iterations(3))

_STATIC_AND_RUNTIME_TEST(
    CONVOLUTION_3D_OP,
    1x1x1_runtime_quant,
    ConvolutionOperatorTester()
        .dimensionality(3)
        .inputSize(8, 9, 10)
        .kernelSize(1, 1, 1)
        .groupInputChannels(5)
        .groupInputChannels(6)
        .iterations(3))

_STATIC_AND_RUNTIME_TEST(
    CONVOLUTION_3D_OP,
    1x1x1_with_qmin,
    ConvolutionOperatorTester()
        .dimensionality(3)
        .inputSize(8, 9, 10)
        .kernelSize(1, 1, 1)
        .groupInputChannels(5)
        .groupInputChannels(6)
        .qmin(128)
        .iterations(3))

_STATIC_AND_RUNTIME_TEST(
    CONVOLUTION_3D_OP,
    1x1x1_with_qmax,
    ConvolutionOperatorTester()
        .dimensionality(3)
        .inputSize(8, 9, 10)
        .kernelSize(1, 1, 1)
        .groupInputChannels(5)
        .groupInputChannels(6)
        .qmax(128)
        .iterations(3))

_STATIC_TEST(
    CONVOLUTION_3D_OP,
    1x1x1_with_input_stride,
    ConvolutionOperatorTester()
        .dimensionality(3)
        .inputSize(8, 9, 10)
        .kernelSize(1, 1, 1)
        .inputPixelStride(28)
        .groupInputChannels(5)
        .groupInputChannels(6)
        .iterations(3))

_STATIC_TEST(
    CONVOLUTION_3D_OP,
    1x1x1_with_output_stride,
    ConvolutionOperatorTester()
        .dimensionality(3)
        .inputSize(8, 9, 10)
        .kernelSize(1, 1, 1)
        .outputPixelStride(7)
        .groupInputChannels(5)
        .groupInputChannels(6)
        .iterations(3))

_STATIC_AND_RUNTIME_TEST(
    CONVOLUTION_3D_OP,
    1x1x1_with_batch,
    ConvolutionOperatorTester()
        .dimensionality(3)
        .inputSize(7, 8, 10)
        .kernelSize(1, 1, 1)
        .batchSize(3)
        .groupInputChannels(5)
        .groupInputChannels(6)
        .iterations(3))

_STATIC_AND_RUNTIME_TEST(
    CONVOLUTION_3D_OP,
    grouped_1x1x1,
    ConvolutionOperatorTester()
        .dimensionality(3)
        .inputSize(8, 10, 11)
        .kernelSize(1, 1, 1)
        .groups(2)
        .groupInputChannels(5)
        .groupInputChannels(6)
        .iterations(3))

TEST(CONVOLUTION_3D_OP, xzp_1x1x1) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  if (pytorch_qnnp_params.q8conv_xzp.kthreshold != SIZE_MAX) {
    ConvolutionOperatorTester()
        .dimensionality(3)
        .inputSize(8, 9, 10)
        .kernelSize(1, 1, 1)
        .groupInputChannels(pytorch_qnnp_params.q8conv_xzp.kthreshold + 1)
        .groupInputChannels(6)
        .iterations(3)
        .testQ8();
  }
}

TEST(CONVOLUTION_3D_OP, xzp_1x1x1_with_qmin) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  if (pytorch_qnnp_params.q8conv_xzp.kthreshold != SIZE_MAX) {
    ConvolutionOperatorTester()
        .dimensionality(3)
        .inputSize(8, 9, 10)
        .kernelSize(1, 1, 1)
        .groupInputChannels(pytorch_qnnp_params.q8conv_xzp.kthreshold + 1)
        .groupInputChannels(6)
        .qmin(128)
        .iterations(3)
        .testQ8();
  }
}

TEST(CONVOLUTION_3D_OP, xzp_1x1x1_with_qmax) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  if (pytorch_qnnp_params.q8conv_xzp.kthreshold != SIZE_MAX) {
    ConvolutionOperatorTester()
        .dimensionality(3)
        .inputSize(8, 9, 10)
        .kernelSize(1, 1, 1)
        .groupInputChannels(pytorch_qnnp_params.q8conv_xzp.kthreshold + 1)
        .groupInputChannels(6)
        .qmax(128)
        .iterations(3)
        .testQ8();
  }
}

TEST(CONVOLUTION_3D_OP, xzp_1x1x1_with_input_stride) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  if (pytorch_qnnp_params.q8conv_xzp.kthreshold != SIZE_MAX) {
    ConvolutionOperatorTester()
        .dimensionality(3)
        .inputSize(8, 9, 10)
        .kernelSize(1, 1, 1)
        .inputPixelStride(pytorch_qnnp_params.q8conv_xzp.kthreshold + 5)
        .groupInputChannels(pytorch_qnnp_params.q8conv_xzp.kthreshold + 1)
        .groupInputChannels(6)
        .iterations(3)
        .testQ8();
  }
}

TEST(CONVOLUTION_3D_OP, xzp_1x1x1_with_output_stride) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  if (pytorch_qnnp_params.q8conv_xzp.kthreshold != SIZE_MAX) {
    ConvolutionOperatorTester()
        .dimensionality(3)
        .inputSize(8, 9, 10)
        .kernelSize(1, 1, 1)
        .outputPixelStride(7)
        .groupInputChannels(pytorch_qnnp_params.q8conv_xzp.kthreshold + 1)
        .groupInputChannels(6)
        .iterations(3)
        .testQ8();
  }
}

TEST(CONVOLUTION_3D_OP, xzp_1x1x1_with_batch) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  if (pytorch_qnnp_params.q8conv_xzp.kthreshold != SIZE_MAX) {
    ConvolutionOperatorTester()
        .dimensionality(3)
        .inputSize(7, 8, 10)
        .kernelSize(1, 1, 1)
        .batchSize(3)
        .groupInputChannels(pytorch_qnnp_params.q8conv_xzp.kthreshold + 1)
        .groupInputChannels(6)
        .iterations(3)
        .testQ8();
  }
}

TEST(CONVOLUTION_3D_OP, grouped_xzp_1x1x1) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  if (pytorch_qnnp_params.q8conv_xzp.kthreshold != SIZE_MAX) {
    ConvolutionOperatorTester()
        .dimensionality(3)
        .inputSize(8, 10, 11)
        .kernelSize(1, 1, 1)
        .groups(2)
        .groupInputChannels(pytorch_qnnp_params.q8conv_xzp.kthreshold + 1)
        .groupInputChannels(6)
        .iterations(3)
        .testQ8();
  }
}

TEST(CONVOLUTION_3D_OP, grouped_xzp_1x1x1_runtime_quant) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  if (pytorch_qnnp_params.q8conv_xzp.kthreshold != SIZE_MAX) {
    ConvolutionOperatorTester()
        .dimensionality(3)
        .inputSize(8, 10, 11)
        .kernelSize(1, 1, 1)
        .groups(2)
        .groupInputChannels(pytorch_qnnp_params.q8conv_xzp.kthreshold + 1)
        .groupInputChannels(6)
        .iterations(3)
        .testQ8(Mode::Runtime);
  }
}

_STATIC_AND_RUNTIME_TEST(
    CONVOLUTION_3D_OP,
    1x1x3,
    ConvolutionOperatorTester()
        .dimensionality(3)
        .inputSize(8, 7, 10)
        .paddingWidth(1)
        .kernelSize(1, 1, 3)
        .groupInputChannels(5)
        .groupInputChannels(6)
        .iterations(3))

_STATIC_AND_RUNTIME_TEST(
    CONVOLUTION_3D_OP,
    grouped_1x1x3,
    ConvolutionOperatorTester()
        .dimensionality(3)
        .inputSize(8, 7, 10)
        .paddingWidth(1)
        .kernelSize(1, 1, 3)
        .groups(2)
        .groupInputChannels(5)
        .groupInputChannels(6)
        .iterations(3))

_STATIC_AND_RUNTIME_TEST(
    CONVOLUTION_3D_OP,
    3x3x1,
    ConvolutionOperatorTester()
        .dimensionality(3)
        .inputSize(8, 9, 7)
        .paddingHeight(1)
        .kernelSize(3, 3, 1)
        .groupInputChannels(5)
        .groupInputChannels(6)
        .iterations(3))

_STATIC_AND_RUNTIME_TEST(
    CONVOLUTION_3D_OP,
    grouped_3x3x1,
    ConvolutionOperatorTester()
        .dimensionality(3)
        .inputSize(8, 9, 7)
        .paddingHeight(1)
        .kernelSize(3, 3, 1)
        .groups(2)
        .groupInputChannels(5)
        .groupInputChannels(6)
        .iterations(3))

_STATIC_AND_RUNTIME_TEST(
    CONVOLUTION_3D_OP,
    3x3x3,
    ConvolutionOperatorTester()
        .dimensionality(3)
        .inputSize(8, 10, 9)
        .padding(1)
        .kernelSize(3, 3, 3)
        .groupInputChannels(5)
        .groupInputChannels(6)
        .iterations(3))

_STATIC_AND_RUNTIME_TEST(
    CONVOLUTION_3D_OP,
    3x3x3_without_padding,
    ConvolutionOperatorTester()
        .dimensionality(3)
        .inputSize(8, 10, 9)
        .kernelSize(3, 3, 3)
        .groupInputChannels(5)
        .groupInputChannels(6)
        .iterations(3))

_STATIC_AND_RUNTIME_TEST(
    CONVOLUTION_3D_OP,
    3x3x3_with_width_padding,
    ConvolutionOperatorTester()
        .dimensionality(3)
        .inputSize(8, 10, 9)
        .paddingWidth(1)
        .kernelSize(3, 3, 3)
        .groupInputChannels(5)
        .groupInputChannels(6)
        .iterations(3))

_STATIC_AND_RUNTIME_TEST(
    CONVOLUTION_3D_OP,
    3x3x3_with_height_padding,
    ConvolutionOperatorTester()
        .dimensionality(3)
        .inputSize(8, 10, 9)
        .paddingHeight(1)
        .kernelSize(3, 3, 3)
        .groupInputChannels(5)
        .groupInputChannels(6)
        .iterations(3))

_STATIC_AND_RUNTIME_TEST(
    CONVOLUTION_3D_OP,
    3x3x3_with_depth_padding,
    ConvolutionOperatorTester()
        .dimensionality(3)
        .inputSize(8, 10, 9)
        .paddingDepth(1)
        .kernelSize(3, 3, 3)
        .groupInputChannels(5)
        .groupInputChannels(6)
        .iterations(3))

_STATIC_TEST(
    CONVOLUTION_3D_OP,
    3x3x3_with_input_stride,
    ConvolutionOperatorTester()
        .dimensionality(3)
        .inputSize(8, 10, 9)
        .padding(1)
        .kernelSize(3, 3, 3)
        .inputPixelStride(22)
        .groupInputChannels(5)
        .groupInputChannels(6)
        .iterations(3))

_STATIC_TEST(
    CONVOLUTION_3D_OP,
    3x3x3_with_output_stride,
    ConvolutionOperatorTester()
        .dimensionality(3)
        .inputSize(8, 10, 9)
        .padding(1)
        .kernelSize(3, 3, 3)
        .outputPixelStride(23)
        .groupInputChannels(5)
        .groupInputChannels(6)
        .iterations(3))

_STATIC_AND_RUNTIME_TEST(
    CONVOLUTION_3D_OP,
    3x3x3_with_batch,
    ConvolutionOperatorTester()
        .dimensionality(3)
        .inputSize(10, 10, 9)
        .padding(1)
        .kernelSize(3, 3, 3)
        .batchSize(3)
        .groupInputChannels(5)
        .groupInputChannels(6)
        .iterations(3))

_STATIC_AND_RUNTIME_TEST(
    CONVOLUTION_3D_OP,
    grouped_3x3x3,
    ConvolutionOperatorTester()
        .dimensionality(3)
        .inputSize(10, 10, 11)
        .padding(1)
        .kernelSize(3, 3, 3)
        .groups(2)
        .groupInputChannels(5)
        .groupInputChannels(6)
        .iterations(3))

_STATIC_AND_RUNTIME_TEST(
    CONVOLUTION_3D_OP,
    3x3x3s2,
    ConvolutionOperatorTester()
        .dimensionality(3)
        .inputSize(8, 10, 12)
        .padding(1)
        .kernelSize(3, 3, 3)
        .subsampling(2)
        .groupInputChannels(5)
        .groupInputChannels(6)
        .iterations(3))

_STATIC_AND_RUNTIME_TEST(
    CONVOLUTION_3D_OP,
    3x3x3s1x2,
    ConvolutionOperatorTester()
        .dimensionality(3)
        .inputSize(8, 9, 10)
        .padding(1)
        .kernelSize(3, 3, 3)
        .subsampling(1, 1, 2)
        .groupInputChannels(5)
        .groupInputChannels(6)
        .iterations(3))

_STATIC_AND_RUNTIME_TEST(
    CONVOLUTION_3D_OP,
    3x3x3s2x1,
    ConvolutionOperatorTester()
        .dimensionality(3)
        .inputSize(8, 9, 10)
        .padding(1)
        .kernelSize(3, 3, 3)
        .subsampling(2, 2, 1)
        .groupInputChannels(5)
        .groupInputChannels(6)
        .iterations(3))

_STATIC_AND_RUNTIME_TEST(
    CONVOLUTION_3D_OP,
    3x3x3d2,
    ConvolutionOperatorTester()
        .dimensionality(3)
        .inputSize(7, 8, 10)
        .padding(2)
        .kernelSize(3, 3, 3)
        .dilation(2)
        .groupInputChannels(5)
        .groupInputChannels(6)
        .iterations(3))

_STATIC_AND_RUNTIME_TEST(
    CONVOLUTION_3D_OP,
    3x3x3d1x2,
    ConvolutionOperatorTester()
        .dimensionality(3)
        .inputSize(11, 10, 8)
        .padding(1, 1, 2)
        .kernelSize(3, 3, 3)
        .dilation(1, 1, 2)
        .groupInputChannels(5)
        .groupInputChannels(6)
        .iterations(3))

_STATIC_AND_RUNTIME_TEST(
    CONVOLUTION_3D_OP,
    3x3x3d2x1,
    ConvolutionOperatorTester()
        .dimensionality(3)
        .inputSize(10, 9, 7)
        .padding(2, 2, 1)
        .kernelSize(3, 3, 3)
        .dilation(2, 2, 1)
        .groupInputChannels(5)
        .groupInputChannels(6)
        .iterations(3))

_STATIC_AND_RUNTIME_TEST(
    CONVOLUTION_3D_OP,
    depthwise_3x3x3,
    ConvolutionOperatorTester()
        .dimensionality(3)
        .inputSize(10, 9, 7)
        .padding(1, 1, 1)
        .kernelSize(3, 3, 3)
        .groups(27)
        .iterations(3))

_STATIC_AND_RUNTIME_TEST(
    CONVOLUTION_3D_OP,
    depthwise_3x3x3s2,
    ConvolutionOperatorTester()
        .dimensionality(3)
        .inputSize(10, 9, 7)
        .padding(1, 1, 1)
        .kernelSize(3, 3, 3)
        .subsampling(2)
        .groups(27)
        .iterations(3))

_STATIC_AND_RUNTIME_TEST(
    CONVOLUTION_3D_OP,
    depthwise_3x3x3s1x2,
    ConvolutionOperatorTester()
        .dimensionality(3)
        .inputSize(10, 9, 7)
        .padding(1, 1, 1)
        .kernelSize(3, 3, 3)
        .subsampling(1, 1, 2)
        .groups(27)
        .iterations(3))

_STATIC_AND_RUNTIME_TEST(
    CONVOLUTION_3D_OP,
    depthwise_3x3x3s2x1,
    ConvolutionOperatorTester()
        .dimensionality(3)
        .inputSize(10, 9, 7)
        .padding(1, 1, 1)
        .kernelSize(3, 3, 3)
        .subsampling(2, 2, 1)
        .groups(27)
        .iterations(3))

_STATIC_AND_RUNTIME_TEST(
    CONVOLUTION_3D_OP,
    depthwise_3x3x3d2,
    ConvolutionOperatorTester()
        .dimensionality(3)
        .inputSize(10, 9, 7)
        .padding(1, 1, 1)
        .kernelSize(3, 3, 3)
        .dilation(2)
        .groups(27)
        .iterations(3))

_STATIC_AND_RUNTIME_TEST(
    CONVOLUTION_3D_OP,
    depthwise_3x3x3d1x2,
    ConvolutionOperatorTester()
        .dimensionality(3)
        .inputSize(10, 9, 7)
        .padding(1, 1, 1)
        .kernelSize(3, 3, 3)
        .dilation(1, 1, 2)
        .groups(27)
        .iterations(3))

_STATIC_AND_RUNTIME_TEST(
    CONVOLUTION_3D_OP,
    depthwise_3x3x3d2x1,
    ConvolutionOperatorTester()
        .dimensionality(3)
        .inputSize(10, 9, 7)
        .padding(1, 1, 1)
        .kernelSize(3, 3, 3)
        .dilation(2, 2, 1)
        .groups(27)
        .iterations(3))

_STATIC_AND_RUNTIME_TEST(
    CONVOLUTION_3D_OP,
    zero_batch_per_channel,
    ConvolutionOperatorTester()
        .dimensionality(3)
        .batchSize(0)
        .inputSize(5, 5, 5)
        .kernelSize(1, 1, 1)
        .groupInputChannels(2)
        .groupOutputChannels(2)
        .iterations(1)
        .per_channel(true))

_STATIC_AND_RUNTIME_TEST(
    CONVOLUTION_3D_OP,
    1x1x1_per_channel,
    ConvolutionOperatorTester()
        .dimensionality(3)
        .inputSize(8, 9, 10)
        .kernelSize(1, 1, 1)
        .groupInputChannels(5)
        .groupInputChannels(6)
        .iterations(3)
        .per_channel(true))

_STATIC_AND_RUNTIME_TEST(
    CONVOLUTION_3D_OP,
    1x1x1_with_qmin_per_channel,
    ConvolutionOperatorTester()
        .dimensionality(3)
        .inputSize(8, 9, 10)
        .kernelSize(1, 1, 1)
        .groupInputChannels(5)
        .groupInputChannels(6)
        .qmin(128)
        .iterations(3)
        .per_channel(true))

_STATIC_AND_RUNTIME_TEST(
    CONVOLUTION_3D_OP,
    1x1x1_with_qmax_per_channel,
    ConvolutionOperatorTester()
        .dimensionality(3)
        .inputSize(8, 9, 10)
        .kernelSize(1, 1, 1)
        .groupInputChannels(5)
        .groupInputChannels(6)
        .qmax(128)
        .iterations(3)
        .per_channel(true))

_STATIC_TEST(
    CONVOLUTION_3D_OP,
    1x1x1_with_input_stride_per_channel,
    ConvolutionOperatorTester()
        .dimensionality(3)
        .inputSize(8, 9, 10)
        .kernelSize(1, 1, 1)
        .inputPixelStride(28)
        .groupInputChannels(5)
        .groupInputChannels(6)
        .iterations(3)
        .per_channel(true))

_STATIC_TEST(
    CONVOLUTION_3D_OP,
    1x1x1_with_output_stride_per_channel,
    ConvolutionOperatorTester()
        .dimensionality(3)
        .inputSize(8, 9, 10)
        .kernelSize(1, 1, 1)
        .outputPixelStride(7)
        .groupInputChannels(5)
        .groupInputChannels(6)
        .iterations(3)
        .per_channel(true))

_STATIC_AND_RUNTIME_TEST(
    CONVOLUTION_3D_OP,
    1x1x1_with_batch_per_channel,
    ConvolutionOperatorTester()
        .dimensionality(3)
        .inputSize(7, 8, 10)
        .kernelSize(1, 1, 1)
        .batchSize(3)
        .groupInputChannels(5)
        .groupInputChannels(6)
        .iterations(3)
        .per_channel(true))

_STATIC_AND_RUNTIME_TEST(
    CONVOLUTION_3D_OP,
    grouped_1x1x1_per_channel,
    ConvolutionOperatorTester()
        .dimensionality(3)
        .inputSize(8, 10, 11)
        .kernelSize(1, 1, 1)
        .groups(2)
        .groupInputChannels(5)
        .groupInputChannels(6)
        .iterations(3)
        .per_channel(true))

TEST(CONVOLUTION_3D_OP, xzp_1x1x1_per_channel) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  if (pytorch_qnnp_params.q8conv_xzp.kthreshold != SIZE_MAX) {
    ConvolutionOperatorTester()
        .dimensionality(3)
        .inputSize(8, 9, 10)
        .kernelSize(1, 1, 1)
        .groupInputChannels(pytorch_qnnp_params.q8conv_xzp.kthreshold + 1)
        .groupInputChannels(6)
        .iterations(3)
        .per_channel(true)
        .testQ8();
  }
}

TEST(CONVOLUTION_3D_OP, xzp_1x1x1_with_qmin_per_channel) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  if (pytorch_qnnp_params.q8conv_xzp.kthreshold != SIZE_MAX) {
    ConvolutionOperatorTester()
        .dimensionality(3)
        .inputSize(8, 9, 10)
        .kernelSize(1, 1, 1)
        .groupInputChannels(pytorch_qnnp_params.q8conv_xzp.kthreshold + 1)
        .groupInputChannels(6)
        .qmin(128)
        .iterations(3)
        .per_channel(true)
        .testQ8();
  }
}

TEST(CONVOLUTION_3D_OP, xzp_1x1x1_with_qmax_per_channel) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  if (pytorch_qnnp_params.q8conv_xzp.kthreshold != SIZE_MAX) {
    ConvolutionOperatorTester()
        .dimensionality(3)
        .inputSize(8, 9, 10)
        .kernelSize(1, 1, 1)
        .groupInputChannels(pytorch_qnnp_params.q8conv_xzp.kthreshold + 1)
        .groupInputChannels(6)
        .qmax(128)
        .iterations(3)
        .per_channel(true)
        .testQ8();
  }
}

TEST(CONVOLUTION_3D_OP, xzp_1x1x1_with_input_stride_per_channel) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  if (pytorch_qnnp_params.q8conv_xzp.kthreshold != SIZE_MAX) {
    ConvolutionOperatorTester()
        .dimensionality(3)
        .inputSize(8, 9, 10)
        .kernelSize(1, 1, 1)
        .inputPixelStride(pytorch_qnnp_params.q8conv_xzp.kthreshold + 5)
        .groupInputChannels(pytorch_qnnp_params.q8conv_xzp.kthreshold + 1)
        .groupInputChannels(6)
        .iterations(3)
        .per_channel(true)
        .testQ8();
  }
}

TEST(CONVOLUTION_3D_OP, xzp_1x1x1_with_output_stride_per_channel) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  if (pytorch_qnnp_params.q8conv_xzp.kthreshold != SIZE_MAX) {
    ConvolutionOperatorTester()
        .dimensionality(3)
        .inputSize(8, 9, 10)
        .kernelSize(1, 1, 1)
        .outputPixelStride(7)
        .groupInputChannels(pytorch_qnnp_params.q8conv_xzp.kthreshold + 1)
        .groupInputChannels(6)
        .iterations(3)
        .per_channel(true)
        .testQ8();
  }
}

TEST(CONVOLUTION_3D_OP, xzp_1x1x1_with_batch_per_channel) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  if (pytorch_qnnp_params.q8conv_xzp.kthreshold != SIZE_MAX) {
    ConvolutionOperatorTester()
        .dimensionality(3)
        .inputSize(7, 8, 10)
        .kernelSize(1, 1, 1)
        .batchSize(3)
        .groupInputChannels(pytorch_qnnp_params.q8conv_xzp.kthreshold + 1)
        .groupInputChannels(6)
        .iterations(3)
        .per_channel(true)
        .testQ8();
  }
}

TEST(CONVOLUTION_3D_OP, grouped_xzp_1x1x1_per_channel) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  if (pytorch_qnnp_params.q8conv_xzp.kthreshold != SIZE_MAX) {
    ConvolutionOperatorTester()
        .dimensionality(3)
        .inputSize(8, 10, 11)
        .kernelSize(1, 1, 1)
        .groups(2)
        .groupInputChannels(pytorch_qnnp_params.q8conv_xzp.kthreshold + 1)
        .groupInputChannels(6)
        .iterations(3)
        .per_channel(true)
        .testQ8();
  }
}

TEST(CONVOLUTION_3D_OP, grouped_xzp_1x1x1_runtime_quant_per_channel) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  if (pytorch_qnnp_params.q8conv_xzp.kthreshold != SIZE_MAX) {
    ConvolutionOperatorTester()
        .dimensionality(3)
        .inputSize(8, 10, 11)
        .kernelSize(1, 1, 1)
        .groups(2)
        .groupInputChannels(pytorch_qnnp_params.q8conv_xzp.kthreshold + 1)
        .groupInputChannels(6)
        .iterations(3)
        .per_channel(true)
        .testQ8(Mode::Runtime);
  }
}

_STATIC_AND_RUNTIME_TEST(
    CONVOLUTION_3D_OP,
    1x1x3_per_channel,
    ConvolutionOperatorTester()
        .dimensionality(3)
        .inputSize(8, 7, 10)
        .paddingWidth(1)
        .kernelSize(1, 1, 3)
        .groupInputChannels(5)
        .groupInputChannels(6)
        .iterations(3)
        .per_channel(true))

_STATIC_AND_RUNTIME_TEST(
    CONVOLUTION_3D_OP,
    grouped_1x1x3_per_channel,
    ConvolutionOperatorTester()
        .dimensionality(3)
        .inputSize(8, 7, 10)
        .paddingWidth(1)
        .kernelSize(1, 1, 3)
        .groups(2)
        .groupInputChannels(5)
        .groupInputChannels(6)
        .iterations(3)
        .per_channel(true))

_STATIC_AND_RUNTIME_TEST(
    CONVOLUTION_3D_OP,
    3x3x1_per_channel,
    ConvolutionOperatorTester()
        .dimensionality(3)
        .inputSize(8, 9, 7)
        .paddingHeight(1)
        .kernelSize(3, 3, 1)
        .groupInputChannels(5)
        .groupInputChannels(6)
        .iterations(3)
        .per_channel(true))

_STATIC_AND_RUNTIME_TEST(
    CONVOLUTION_3D_OP,
    grouped_3x3x1_per_channel,
    ConvolutionOperatorTester()
        .dimensionality(3)
        .inputSize(8, 9, 7)
        .paddingHeight(1)
        .kernelSize(3, 3, 1)
        .groups(2)
        .groupInputChannels(5)
        .groupInputChannels(6)
        .iterations(3)
        .per_channel(true))

_STATIC_AND_RUNTIME_TEST(
    CONVOLUTION_3D_OP,
    3x3x3_per_channel,
    ConvolutionOperatorTester()
        .dimensionality(3)
        .inputSize(8, 10, 9)
        .padding(1)
        .kernelSize(3, 3, 3)
        .groupInputChannels(5)
        .groupInputChannels(6)
        .iterations(3)
        .per_channel(true))

_STATIC_AND_RUNTIME_TEST(
    CONVOLUTION_3D_OP,
    3x3x3_without_padding_per_channel,
    ConvolutionOperatorTester()
        .dimensionality(3)
        .inputSize(8, 10, 9)
        .kernelSize(3, 3, 3)
        .groupInputChannels(5)
        .groupInputChannels(6)
        .iterations(3)
        .per_channel(true))

_STATIC_AND_RUNTIME_TEST(
    CONVOLUTION_3D_OP,
    3x3x3_with_width_padding_per_channel,
    ConvolutionOperatorTester()
        .dimensionality(3)
        .inputSize(8, 10, 9)
        .paddingWidth(1)
        .kernelSize(3, 3, 3)
        .groupInputChannels(5)
        .groupInputChannels(6)
        .iterations(3)
        .per_channel(true))

_STATIC_AND_RUNTIME_TEST(
    CONVOLUTION_3D_OP,
    3x3x3_with_height_padding_per_channel,
    ConvolutionOperatorTester()
        .dimensionality(3)
        .inputSize(8, 10, 9)
        .paddingHeight(1)
        .kernelSize(3, 3, 3)
        .groupInputChannels(5)
        .groupInputChannels(6)
        .iterations(3)
        .per_channel(true))

_STATIC_AND_RUNTIME_TEST(
    CONVOLUTION_3D_OP,
    3x3x3_with_depth_padding_per_channel,
    ConvolutionOperatorTester()
        .dimensionality(3)
        .inputSize(8, 10, 9)
        .paddingDepth(1)
        .kernelSize(3, 3, 3)
        .groupInputChannels(5)
        .groupInputChannels(6)
        .iterations(3)
        .per_channel(true))

_STATIC_TEST(
    CONVOLUTION_3D_OP,
    3x3x3_with_input_stride_per_channel,
    ConvolutionOperatorTester()
        .dimensionality(3)
        .inputSize(8, 10, 9)
        .padding(1)
        .kernelSize(3, 3, 3)
        .inputPixelStride(22)
        .groupInputChannels(5)
        .groupInputChannels(6)
        .iterations(3)
        .per_channel(true))

_STATIC_TEST(
    CONVOLUTION_3D_OP,
    3x3x3_with_output_stride_per_channel,
    ConvolutionOperatorTester()
        .dimensionality(3)
        .inputSize(8, 10, 9)
        .padding(1)
        .kernelSize(3, 3, 3)
        .outputPixelStride(23)
        .groupInputChannels(5)
        .groupInputChannels(6)
        .iterations(3)
        .per_channel(true))

_STATIC_AND_RUNTIME_TEST(
    CONVOLUTION_3D_OP,
    3x3x3_with_batch_per_channel,
    ConvolutionOperatorTester()
        .dimensionality(3)
        .inputSize(10, 10, 9)
        .padding(1)
        .kernelSize(3, 3, 3)
        .batchSize(3)
        .groupInputChannels(5)
        .groupInputChannels(6)
        .iterations(3)
        .per_channel(true))

_STATIC_AND_RUNTIME_TEST(
    CONVOLUTION_3D_OP,
    grouped_3x3x3_per_channel,
    ConvolutionOperatorTester()
        .dimensionality(3)
        .inputSize(10, 10, 11)
        .padding(1)
        .kernelSize(3, 3, 3)
        .groups(2)
        .groupInputChannels(5)
        .groupInputChannels(6)
        .iterations(3)
        .per_channel(true))

_STATIC_AND_RUNTIME_TEST(
    CONVOLUTION_3D_OP,
    3x3x3s2_per_channel,
    ConvolutionOperatorTester()
        .dimensionality(3)
        .inputSize(8, 10, 12)
        .padding(1)
        .kernelSize(3, 3, 3)
        .subsampling(2)
        .groupInputChannels(5)
        .groupInputChannels(6)
        .iterations(3)
        .per_channel(true))

_STATIC_AND_RUNTIME_TEST(
    CONVOLUTION_3D_OP,
    3x3x3s1x2_per_channel,
    ConvolutionOperatorTester()
        .dimensionality(3)
        .inputSize(8, 9, 10)
        .padding(1)
        .kernelSize(3, 3, 3)
        .subsampling(1, 1, 2)
        .groupInputChannels(5)
        .groupInputChannels(6)
        .iterations(3)
        .per_channel(true))

_STATIC_AND_RUNTIME_TEST(
    CONVOLUTION_3D_OP,
    3x3x3s2x1_per_channel,
    ConvolutionOperatorTester()
        .dimensionality(3)
        .inputSize(8, 9, 10)
        .padding(1)
        .kernelSize(3, 3, 3)
        .subsampling(2, 2, 1)
        .groupInputChannels(5)
        .groupInputChannels(6)
        .iterations(3)
        .per_channel(true))

_STATIC_AND_RUNTIME_TEST(
    CONVOLUTION_3D_OP,
    3x3x3d2_per_channel,
    ConvolutionOperatorTester()
        .dimensionality(3)
        .inputSize(7, 8, 10)
        .padding(2)
        .kernelSize(3, 3, 3)
        .dilation(2)
        .groupInputChannels(5)
        .groupInputChannels(6)
        .iterations(3)
        .per_channel(true))

_STATIC_AND_RUNTIME_TEST(
    CONVOLUTION_3D_OP,
    3x3x3d1x2_per_channel,
    ConvolutionOperatorTester()
        .dimensionality(3)
        .inputSize(11, 10, 8)
        .padding(1, 1, 2)
        .kernelSize(3, 3, 3)
        .dilation(1, 1, 2)
        .groupInputChannels(5)
        .groupInputChannels(6)
        .iterations(3)
        .per_channel(true))

_STATIC_AND_RUNTIME_TEST(
    CONVOLUTION_3D_OP,
    3x3x3d2x1_per_channel,
    ConvolutionOperatorTester()
        .dimensionality(3)
        .inputSize(10, 9, 7)
        .padding(2, 2, 1)
        .kernelSize(3, 3, 3)
        .dilation(2, 2, 1)
        .groupInputChannels(5)
        .groupInputChannels(6)
        .iterations(3)
        .per_channel(true))

_STATIC_AND_RUNTIME_TEST(
    CONVOLUTION_3D_OP,
    depthwise_3x3x3_per_channel,
    ConvolutionOperatorTester()
        .dimensionality(3)
        .inputSize(10, 9, 7)
        .padding(1, 1, 1)
        .kernelSize(3, 3, 3)
        .groups(27)
        .iterations(3)
        .per_channel(true))

_STATIC_AND_RUNTIME_TEST(
    CONVOLUTION_3D_OP,
    depthwise_3x3x3s2_per_channel,
    ConvolutionOperatorTester()
        .dimensionality(3)
        .inputSize(10, 9, 7)
        .padding(1, 1, 1)
        .kernelSize(3, 3, 3)
        .subsampling(2)
        .groups(27)
        .iterations(3)
        .per_channel(true))

_STATIC_AND_RUNTIME_TEST(
    CONVOLUTION_3D_OP,
    depthwise_3x3x3s1x2_per_channel,
    ConvolutionOperatorTester()
        .dimensionality(3)
        .inputSize(10, 9, 7)
        .padding(1, 1, 1)
        .kernelSize(3, 3, 3)
        .subsampling(1, 1, 2)
        .groups(27)
        .iterations(3)
        .per_channel(true))

_STATIC_AND_RUNTIME_TEST(
    CONVOLUTION_3D_OP,
    depthwise_3x3x3s2x1_per_channel,
    ConvolutionOperatorTester()
        .dimensionality(3)
        .inputSize(10, 9, 7)
        .padding(1, 1, 1)
        .kernelSize(3, 3, 3)
        .subsampling(2, 2, 1)
        .groups(27)
        .iterations(3)
        .per_channel(true))

_STATIC_AND_RUNTIME_TEST(
    CONVOLUTION_3D_OP,
    depthwise_3x3x3d2_per_channel,
    ConvolutionOperatorTester()
        .dimensionality(3)
        .inputSize(10, 9, 7)
        .padding(1, 1, 1)
        .kernelSize(3, 3, 3)
        .dilation(2)
        .groups(27)
        .iterations(3)
        .per_channel(true))

_STATIC_AND_RUNTIME_TEST(
    CONVOLUTION_3D_OP,
    depthwise_3x3x3d1x2_per_channel,
    ConvolutionOperatorTester()
        .dimensionality(3)
        .inputSize(10, 9, 7)
        .padding(1, 1, 1)
        .kernelSize(3, 3, 3)
        .dilation(1, 1, 2)
        .groups(27)
        .iterations(3)
        .per_channel(true))

_STATIC_AND_RUNTIME_TEST(
    CONVOLUTION_3D_OP,
    depthwise_3x3x3d2x1_per_channel,
    ConvolutionOperatorTester()
        .dimensionality(3)
        .inputSize(10, 9, 7)
        .padding(1, 1, 1)
        .kernelSize(3, 3, 3)
        .dilation(2, 2, 1)
        .groups(27)
        .iterations(3)
        .per_channel(true))
