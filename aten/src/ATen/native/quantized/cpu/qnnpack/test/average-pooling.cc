/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <qnnpack/params.h>

#include "average-pooling-operator-tester.h"

TEST(AVERAGE_POOLING_OP, zero_batch) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  AveragePoolingOperatorTester()
      .batchSize(0)
      .inputHeight(2)
      .inputWidth(4)
      .poolingHeight(1)
      .poolingWidth(2)
      .channels(4)
      .testQ8();
}

TEST(AVERAGE_POOLING_OP, unit_batch_many_channels_small_1xM_pool) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.q8avgpool.kr;
       channels <= 3 * pytorch_qnnp_params.q8avgpool.kr;
       channels++) {
    for (size_t poolSize = 2; poolSize <= pytorch_qnnp_params.q8avgpool.mr;
         poolSize++) {
      AveragePoolingOperatorTester()
          .batchSize(1)
          .inputHeight(2)
          .inputWidth(poolSize + 2)
          .poolingHeight(1)
          .poolingWidth(poolSize)
          .channels(channels)
          .testQ8();
    }
  }
}

TEST(AVERAGE_POOLING_OP, unit_batch_many_channels_small_1xM_pool_with_padding) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.q8avgpool.kr;
       channels <= 3 * pytorch_qnnp_params.q8avgpool.kr;
       channels += 3) {
    for (size_t poolSize = 3; poolSize <= pytorch_qnnp_params.q8avgpool.mr;
         poolSize++) {
      for (size_t paddingWidth = 0; paddingWidth <= 1; paddingWidth++) {
        AveragePoolingOperatorTester()
            .batchSize(1)
            .inputHeight(2)
            .inputWidth(poolSize + 2)
            .paddingWidth(paddingWidth)
            .poolingHeight(1)
            .poolingWidth(poolSize)
            .channels(channels)
            .testQ8();
      }
    }
  }
}

TEST(AVERAGE_POOLING_OP, unit_batch_many_channels_small_1xM_pool_with_stride) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.q8avgpool.kr;
       channels <= 3 * pytorch_qnnp_params.q8avgpool.kr;
       channels += 3) {
    for (size_t poolSize = 2; poolSize <= pytorch_qnnp_params.q8avgpool.mr;
         poolSize++) {
      AveragePoolingOperatorTester()
          .batchSize(1)
          .inputHeight(2)
          .inputWidth(poolSize + 4)
          .poolingHeight(1)
          .poolingWidth(poolSize)
          .strideWidth(2)
          .channels(channels)
          .testQ8();
    }
  }
}

TEST(AVERAGE_POOLING_OP, unit_batch_many_channels_small_Mx1_pool) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.q8avgpool.kr;
       channels <= 3 * pytorch_qnnp_params.q8avgpool.kr;
       channels++) {
    for (size_t poolSize = 2; poolSize <= pytorch_qnnp_params.q8avgpool.mr;
         poolSize++) {
      AveragePoolingOperatorTester()
          .batchSize(1)
          .inputHeight(poolSize + 1)
          .inputWidth(3)
          .poolingHeight(poolSize)
          .poolingWidth(1)
          .channels(channels)
          .testQ8();
    }
  }
}

TEST(AVERAGE_POOLING_OP, unit_batch_many_channels_small_Mx1_pool_with_padding) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.q8avgpool.kr;
       channels <= 3 * pytorch_qnnp_params.q8avgpool.kr;
       channels += 3) {
    for (size_t poolSize = 2; poolSize <= pytorch_qnnp_params.q8avgpool.mr;
         poolSize++) {
      for (size_t paddingHeight = 0; paddingHeight <= 1; paddingHeight++) {
        AveragePoolingOperatorTester()
            .batchSize(1)
            .inputHeight(poolSize + 1)
            .inputWidth(3)
            .paddingHeight(paddingHeight)
            .poolingHeight(poolSize)
            .poolingWidth(1)
            .channels(channels)
            .testQ8();
      }
    }
  }
}

TEST(AVERAGE_POOLING_OP, unit_batch_many_channels_small_Mx1_pool_with_stride) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.q8avgpool.kr;
       channels <= 3 * pytorch_qnnp_params.q8avgpool.kr;
       channels += 3) {
    for (size_t poolSize = 2; poolSize <= pytorch_qnnp_params.q8avgpool.mr;
         poolSize++) {
      AveragePoolingOperatorTester()
          .batchSize(1)
          .inputHeight(poolSize + 3)
          .inputWidth(3)
          .poolingHeight(poolSize)
          .poolingWidth(1)
          .strideHeight(2)
          .channels(channels)
          .testQ8();
    }
  }
}

TEST(
    AVERAGE_POOLING_OP,
    unit_batch_many_channels_small_pool_with_input_stride) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.q8avgpool.kr;
       channels <= 3 * pytorch_qnnp_params.q8avgpool.kr;
       channels += 3) {
    for (size_t poolSize = 2; poolSize <= pytorch_qnnp_params.q8avgpool.mr;
         poolSize++) {
      AveragePoolingOperatorTester()
          .batchSize(1)
          .inputHeight(poolSize + 1)
          .inputWidth(3)
          .poolingHeight(poolSize)
          .poolingWidth(1)
          .channels(channels)
          .inputPixelStride(5 * pytorch_qnnp_params.q8avgpool.kr)
          .testQ8();
      AveragePoolingOperatorTester()
          .batchSize(1)
          .inputHeight(2)
          .inputWidth(poolSize + 2)
          .poolingHeight(1)
          .poolingWidth(poolSize)
          .channels(channels)
          .inputPixelStride(5 * pytorch_qnnp_params.q8avgpool.kr)
          .testQ8();
    }
  }
}

TEST(
    AVERAGE_POOLING_OP,
    unit_batch_many_channels_small_pool_with_output_stride) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.q8avgpool.kr;
       channels <= 3 * pytorch_qnnp_params.q8avgpool.kr;
       channels += 3) {
    for (size_t poolSize = 2; poolSize <= pytorch_qnnp_params.q8avgpool.mr;
         poolSize++) {
      AveragePoolingOperatorTester()
          .batchSize(1)
          .inputHeight(poolSize + 1)
          .inputWidth(3)
          .poolingHeight(poolSize)
          .poolingWidth(1)
          .channels(channels)
          .outputPixelStride(5 * pytorch_qnnp_params.q8avgpool.kr)
          .testQ8();
      AveragePoolingOperatorTester()
          .batchSize(1)
          .inputHeight(2)
          .inputWidth(poolSize + 2)
          .poolingHeight(1)
          .poolingWidth(poolSize)
          .channels(channels)
          .outputPixelStride(5 * pytorch_qnnp_params.q8avgpool.kr)
          .testQ8();
    }
  }
}

TEST(AVERAGE_POOLING_OP, unit_batch_many_channels_small_pool_with_input_scale) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.q8avgpool.kr;
       channels <= 3 * pytorch_qnnp_params.q8avgpool.kr;
       channels += 3) {
    for (size_t poolSize = 2; poolSize <= pytorch_qnnp_params.q8avgpool.mr;
         poolSize++) {
      for (float inputScale = 0.01f; inputScale < 100.0f;
           inputScale *= 3.14159265f) {
        AveragePoolingOperatorTester()
            .batchSize(1)
            .inputHeight(poolSize + 1)
            .inputWidth(3)
            .poolingHeight(poolSize)
            .poolingWidth(1)
            .channels(channels)
            .inputScale(inputScale)
            .testQ8();
        AveragePoolingOperatorTester()
            .batchSize(1)
            .inputHeight(2)
            .inputWidth(poolSize + 2)
            .poolingHeight(1)
            .poolingWidth(poolSize)
            .channels(channels)
            .inputScale(inputScale)
            .testQ8();
      }
    }
  }
}

TEST(
    AVERAGE_POOLING_OP,
    unit_batch_many_channels_small_pool_with_input_zero_point) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.q8avgpool.kr;
       channels <= 3 * pytorch_qnnp_params.q8avgpool.kr;
       channels += 3) {
    for (size_t poolSize = 2; poolSize <= pytorch_qnnp_params.q8avgpool.mr;
         poolSize++) {
      for (int32_t inputZeroPoint = 0; inputZeroPoint <= 255;
           inputZeroPoint += 51) {
        AveragePoolingOperatorTester()
            .batchSize(1)
            .inputHeight(poolSize + 1)
            .inputWidth(3)
            .poolingHeight(poolSize)
            .poolingWidth(1)
            .channels(channels)
            .inputZeroPoint(uint8_t(inputZeroPoint))
            .testQ8();
        AveragePoolingOperatorTester()
            .batchSize(1)
            .inputHeight(2)
            .inputWidth(poolSize + 2)
            .poolingHeight(1)
            .poolingWidth(poolSize)
            .channels(channels)
            .inputZeroPoint(uint8_t(inputZeroPoint))
            .testQ8();
      }
    }
  }
}

TEST(
    AVERAGE_POOLING_OP,
    unit_batch_many_channels_small_pool_with_output_scale) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.q8avgpool.kr;
       channels <= 3 * pytorch_qnnp_params.q8avgpool.kr;
       channels += 3) {
    for (size_t poolSize = 2; poolSize <= pytorch_qnnp_params.q8avgpool.mr;
         poolSize++) {
      for (float outputScale = 0.01f; outputScale < 100.0f;
           outputScale *= 3.14159265f) {
        AveragePoolingOperatorTester()
            .batchSize(1)
            .inputHeight(poolSize + 1)
            .inputWidth(3)
            .poolingHeight(poolSize)
            .poolingWidth(1)
            .channels(channels)
            .outputScale(outputScale)
            .testQ8();
        AveragePoolingOperatorTester()
            .batchSize(1)
            .inputHeight(2)
            .inputWidth(poolSize + 2)
            .poolingHeight(1)
            .poolingWidth(poolSize)
            .channels(channels)
            .outputScale(outputScale)
            .testQ8();
      }
    }
  }
}

TEST(
    AVERAGE_POOLING_OP,
    unit_batch_many_channels_small_pool_with_output_zero_point) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.q8avgpool.kr;
       channels <= 3 * pytorch_qnnp_params.q8avgpool.kr;
       channels += 3) {
    for (size_t poolSize = 2; poolSize <= pytorch_qnnp_params.q8avgpool.mr;
         poolSize++) {
      for (int32_t outputZeroPoint = 0; outputZeroPoint <= 255;
           outputZeroPoint += 51) {
        AveragePoolingOperatorTester()
            .batchSize(1)
            .inputHeight(poolSize + 1)
            .inputWidth(3)
            .poolingHeight(poolSize)
            .poolingWidth(1)
            .channels(channels)
            .outputZeroPoint(uint8_t(outputZeroPoint))
            .testQ8();
        AveragePoolingOperatorTester()
            .batchSize(1)
            .inputHeight(2)
            .inputWidth(poolSize + 2)
            .poolingHeight(1)
            .poolingWidth(poolSize)
            .channels(channels)
            .outputZeroPoint(uint8_t(outputZeroPoint))
            .testQ8();
      }
    }
  }
}

TEST(AVERAGE_POOLING_OP, unit_batch_many_channels_small_pool_with_qmin) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.q8avgpool.kr;
       channels <= 3 * pytorch_qnnp_params.q8avgpool.kr;
       channels += 3) {
    for (size_t poolSize = 2; poolSize <= pytorch_qnnp_params.q8avgpool.mr;
         poolSize++) {
      AveragePoolingOperatorTester()
          .batchSize(1)
          .inputHeight(poolSize + 1)
          .inputWidth(3)
          .poolingHeight(poolSize)
          .poolingWidth(1)
          .channels(channels)
          .qmin(128)
          .testQ8();
      AveragePoolingOperatorTester()
          .batchSize(1)
          .inputHeight(2)
          .inputWidth(poolSize + 2)
          .poolingHeight(1)
          .poolingWidth(poolSize)
          .channels(channels)
          .qmin(128)
          .testQ8();
    }
  }
}

TEST(AVERAGE_POOLING_OP, unit_batch_many_channels_small_pool_with_qmax) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.q8avgpool.kr;
       channels <= 3 * pytorch_qnnp_params.q8avgpool.kr;
       channels += 3) {
    for (size_t poolSize = 2; poolSize <= pytorch_qnnp_params.q8avgpool.mr;
         poolSize++) {
      AveragePoolingOperatorTester()
          .batchSize(1)
          .inputHeight(poolSize + 1)
          .inputWidth(3)
          .poolingHeight(poolSize)
          .poolingWidth(1)
          .channels(channels)
          .qmax(128)
          .testQ8();
      AveragePoolingOperatorTester()
          .batchSize(1)
          .inputHeight(2)
          .inputWidth(poolSize + 2)
          .poolingHeight(1)
          .poolingWidth(poolSize)
          .channels(channels)
          .qmax(128)
          .testQ8();
    }
  }
}

TEST(AVERAGE_POOLING_OP, unit_batch_many_channels_large_1xM_pool) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.q8avgpool.kr;
       channels <= 3 * pytorch_qnnp_params.q8avgpool.kr;
       channels++) {
    for (size_t poolSize = pytorch_qnnp_params.q8avgpool.mr + 1; poolSize <=
         pytorch_qnnp_params.q8avgpool.mr + pytorch_qnnp_params.q8avgpool.qr;
         poolSize++) {
      AveragePoolingOperatorTester()
          .batchSize(1)
          .inputHeight(2)
          .inputWidth(poolSize + 2)
          .poolingHeight(1)
          .poolingWidth(poolSize)
          .channels(channels)
          .testQ8();
    }
  }
}

TEST(AVERAGE_POOLING_OP, unit_batch_many_channels_large_1xM_pool_with_padding) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.q8avgpool.kr;
       channels <= 3 * pytorch_qnnp_params.q8avgpool.kr;
       channels += 3) {
    for (size_t poolSize = 3; poolSize <= pytorch_qnnp_params.q8avgpool.mr;
         poolSize++) {
      for (size_t paddingWidth = 0; paddingWidth <= 1; paddingWidth++) {
        AveragePoolingOperatorTester()
            .batchSize(1)
            .inputHeight(2)
            .inputWidth(poolSize + 2)
            .paddingWidth(paddingWidth)
            .poolingHeight(1)
            .poolingWidth(poolSize)
            .channels(channels)
            .testQ8();
      }
    }
  }
}

TEST(AVERAGE_POOLING_OP, unit_batch_many_channels_large_1xM_pool_with_stride) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.q8avgpool.kr;
       channels <= 3 * pytorch_qnnp_params.q8avgpool.kr;
       channels++) {
    for (size_t poolSize = pytorch_qnnp_params.q8avgpool.mr + 1; poolSize <=
         pytorch_qnnp_params.q8avgpool.mr + pytorch_qnnp_params.q8avgpool.qr;
         poolSize++) {
      AveragePoolingOperatorTester()
          .batchSize(1)
          .inputHeight(2)
          .inputWidth(poolSize + 4)
          .poolingHeight(1)
          .poolingWidth(poolSize)
          .strideWidth(2)
          .channels(channels)
          .testQ8();
    }
  }
}

TEST(AVERAGE_POOLING_OP, unit_batch_many_channels_large_Mx1_pool) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.q8avgpool.kr;
       channels <= 3 * pytorch_qnnp_params.q8avgpool.kr;
       channels++) {
    for (size_t poolSize = pytorch_qnnp_params.q8avgpool.mr + 1; poolSize <=
         pytorch_qnnp_params.q8avgpool.mr + pytorch_qnnp_params.q8avgpool.qr;
         poolSize++) {
      AveragePoolingOperatorTester()
          .batchSize(1)
          .inputHeight(poolSize + 1)
          .inputWidth(3)
          .poolingHeight(poolSize)
          .poolingWidth(1)
          .channels(channels)
          .testQ8();
    }
  }
}

TEST(AVERAGE_POOLING_OP, unit_batch_many_channels_large_Mx1_pool_with_padding) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.q8avgpool.kr;
       channels <= 3 * pytorch_qnnp_params.q8avgpool.kr;
       channels += 3) {
    for (size_t poolSize = pytorch_qnnp_params.q8avgpool.mr + 1; poolSize <=
         pytorch_qnnp_params.q8avgpool.mr + pytorch_qnnp_params.q8avgpool.qr;
         poolSize++) {
      for (size_t paddingHeight = 0; paddingHeight <= 1; paddingHeight++) {
        AveragePoolingOperatorTester()
            .batchSize(1)
            .inputHeight(poolSize + 1)
            .inputWidth(3)
            .paddingHeight(paddingHeight)
            .poolingHeight(poolSize)
            .poolingWidth(1)
            .channels(channels)
            .testQ8();
      }
    }
  }
}

TEST(AVERAGE_POOLING_OP, unit_batch_many_channels_large_Mx1_pool_with_stride) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.q8avgpool.kr;
       channels <= 3 * pytorch_qnnp_params.q8avgpool.kr;
       channels += 3) {
    for (size_t poolSize = pytorch_qnnp_params.q8avgpool.mr + 1; poolSize <=
         pytorch_qnnp_params.q8avgpool.mr + pytorch_qnnp_params.q8avgpool.qr;
         poolSize++) {
      for (size_t paddingHeight = 0; paddingHeight <= 1; paddingHeight++) {
        AveragePoolingOperatorTester()
            .batchSize(1)
            .inputHeight(poolSize + 1)
            .inputWidth(3)
            .paddingHeight(paddingHeight)
            .poolingHeight(poolSize)
            .poolingWidth(1)
            .channels(channels)
            .testQ8();
      }
    }
  }
}

TEST(
    AVERAGE_POOLING_OP,
    unit_batch_many_channels_large_pool_with_input_stride) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.q8avgpool.kr;
       channels <= 3 * pytorch_qnnp_params.q8avgpool.kr;
       channels++) {
    for (size_t poolSize = pytorch_qnnp_params.q8avgpool.mr + 1; poolSize <=
         pytorch_qnnp_params.q8avgpool.mr + pytorch_qnnp_params.q8avgpool.qr;
         poolSize++) {
      AveragePoolingOperatorTester()
          .batchSize(1)
          .inputHeight(poolSize + 1)
          .inputWidth(3)
          .poolingHeight(poolSize)
          .poolingWidth(1)
          .channels(channels)
          .inputPixelStride(5 * pytorch_qnnp_params.q8avgpool.kr)
          .testQ8();
      AveragePoolingOperatorTester()
          .batchSize(1)
          .inputHeight(2)
          .inputWidth(poolSize + 2)
          .poolingHeight(1)
          .poolingWidth(poolSize)
          .channels(channels)
          .inputPixelStride(5 * pytorch_qnnp_params.q8avgpool.kr)
          .testQ8();
    }
  }
}

TEST(AVERAGE_POOLING_OP, unit_batch_many_channels_large_pool_with_input_scale) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.q8avgpool.kr;
       channels <= 3 * pytorch_qnnp_params.q8avgpool.kr;
       channels += 3) {
    for (size_t poolSize = pytorch_qnnp_params.q8avgpool.mr + 1; poolSize <=
         pytorch_qnnp_params.q8avgpool.mr + pytorch_qnnp_params.q8avgpool.qr;
         poolSize++) {
      for (float inputScale = 0.01f; inputScale < 100.0f;
           inputScale *= 3.14159265f) {
        AveragePoolingOperatorTester()
            .batchSize(1)
            .inputHeight(poolSize + 1)
            .inputWidth(3)
            .poolingHeight(poolSize)
            .poolingWidth(1)
            .channels(channels)
            .inputScale(inputScale)
            .testQ8();
        AveragePoolingOperatorTester()
            .batchSize(1)
            .inputHeight(2)
            .inputWidth(poolSize + 2)
            .poolingHeight(1)
            .poolingWidth(poolSize)
            .channels(channels)
            .inputScale(inputScale)
            .testQ8();
      }
    }
  }
}

TEST(
    AVERAGE_POOLING_OP,
    unit_batch_many_channels_large_pool_with_input_zero_point) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.q8avgpool.kr;
       channels <= 3 * pytorch_qnnp_params.q8avgpool.kr;
       channels += 3) {
    for (size_t poolSize = pytorch_qnnp_params.q8avgpool.mr + 1; poolSize <=
         pytorch_qnnp_params.q8avgpool.mr + pytorch_qnnp_params.q8avgpool.qr;
         poolSize++) {
      for (int32_t inputZeroPoint = 0; inputZeroPoint <= 255;
           inputZeroPoint += 51) {
        AveragePoolingOperatorTester()
            .batchSize(1)
            .inputHeight(poolSize + 1)
            .inputWidth(3)
            .poolingHeight(poolSize)
            .poolingWidth(1)
            .channels(channels)
            .inputZeroPoint(uint8_t(inputZeroPoint))
            .testQ8();
        AveragePoolingOperatorTester()
            .batchSize(1)
            .inputHeight(2)
            .inputWidth(poolSize + 2)
            .poolingHeight(1)
            .poolingWidth(poolSize)
            .channels(channels)
            .inputZeroPoint(uint8_t(inputZeroPoint))
            .testQ8();
      }
    }
  }
}

TEST(
    AVERAGE_POOLING_OP,
    unit_batch_many_channels_large_pool_with_output_stride) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.q8avgpool.kr;
       channels <= 3 * pytorch_qnnp_params.q8avgpool.kr;
       channels++) {
    for (size_t poolSize = pytorch_qnnp_params.q8avgpool.mr + 1; poolSize <=
         pytorch_qnnp_params.q8avgpool.mr + pytorch_qnnp_params.q8avgpool.qr;
         poolSize++) {
      AveragePoolingOperatorTester()
          .batchSize(1)
          .inputHeight(poolSize + 1)
          .inputWidth(3)
          .poolingHeight(poolSize)
          .poolingWidth(1)
          .channels(channels)
          .outputPixelStride(5 * pytorch_qnnp_params.q8avgpool.kr)
          .testQ8();
      AveragePoolingOperatorTester()
          .batchSize(1)
          .inputHeight(2)
          .inputWidth(poolSize + 2)
          .poolingHeight(1)
          .poolingWidth(poolSize)
          .channels(channels)
          .outputPixelStride(5 * pytorch_qnnp_params.q8avgpool.kr)
          .testQ8();
    }
  }
}

TEST(
    AVERAGE_POOLING_OP,
    unit_batch_many_channels_large_pool_with_output_scale) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.q8avgpool.kr;
       channels <= 3 * pytorch_qnnp_params.q8avgpool.kr;
       channels += 3) {
    for (size_t poolSize = pytorch_qnnp_params.q8avgpool.mr + 1; poolSize <=
         pytorch_qnnp_params.q8avgpool.mr + pytorch_qnnp_params.q8avgpool.qr;
         poolSize++) {
      for (float outputScale = 0.01f; outputScale < 100.0f;
           outputScale *= 3.14159265f) {
        AveragePoolingOperatorTester()
            .batchSize(1)
            .inputHeight(poolSize + 1)
            .inputWidth(3)
            .poolingHeight(poolSize)
            .poolingWidth(1)
            .channels(channels)
            .outputScale(outputScale)
            .testQ8();
        AveragePoolingOperatorTester()
            .batchSize(1)
            .inputHeight(2)
            .inputWidth(poolSize + 2)
            .poolingHeight(1)
            .poolingWidth(poolSize)
            .channels(channels)
            .outputScale(outputScale)
            .testQ8();
      }
    }
  }
}

TEST(
    AVERAGE_POOLING_OP,
    unit_batch_many_channels_large_pool_with_output_zero_point) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.q8avgpool.kr;
       channels <= 3 * pytorch_qnnp_params.q8avgpool.kr;
       channels += 3) {
    for (size_t poolSize = pytorch_qnnp_params.q8avgpool.mr + 1; poolSize <=
         pytorch_qnnp_params.q8avgpool.mr + pytorch_qnnp_params.q8avgpool.qr;
         poolSize++) {
      for (int32_t outputZeroPoint = 0; outputZeroPoint <= 255;
           outputZeroPoint += 51) {
        AveragePoolingOperatorTester()
            .batchSize(1)
            .inputHeight(poolSize + 1)
            .inputWidth(3)
            .poolingHeight(poolSize)
            .poolingWidth(1)
            .channels(channels)
            .outputZeroPoint(uint8_t(outputZeroPoint))
            .testQ8();
        AveragePoolingOperatorTester()
            .batchSize(1)
            .inputHeight(2)
            .inputWidth(poolSize + 2)
            .poolingHeight(1)
            .poolingWidth(poolSize)
            .channels(channels)
            .outputZeroPoint(uint8_t(outputZeroPoint))
            .testQ8();
      }
    }
  }
}

TEST(AVERAGE_POOLING_OP, unit_batch_many_channels_large_pool_with_qmin) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.q8avgpool.kr;
       channels <= 3 * pytorch_qnnp_params.q8avgpool.kr;
       channels += 3) {
    for (size_t poolSize = pytorch_qnnp_params.q8avgpool.mr + 1; poolSize <=
         pytorch_qnnp_params.q8avgpool.mr + pytorch_qnnp_params.q8avgpool.qr;
         poolSize++) {
      AveragePoolingOperatorTester()
          .batchSize(1)
          .inputHeight(poolSize + 1)
          .inputWidth(3)
          .poolingHeight(poolSize)
          .poolingWidth(1)
          .channels(channels)
          .qmin(128)
          .testQ8();
      AveragePoolingOperatorTester()
          .batchSize(1)
          .inputHeight(2)
          .inputWidth(poolSize + 2)
          .poolingHeight(1)
          .poolingWidth(poolSize)
          .channels(channels)
          .qmin(128)
          .testQ8();
    }
  }
}

TEST(AVERAGE_POOLING_OP, unit_batch_many_channels_large_pool_with_qmax) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.q8avgpool.kr;
       channels <= 3 * pytorch_qnnp_params.q8avgpool.kr;
       channels += 3) {
    for (size_t poolSize = pytorch_qnnp_params.q8avgpool.mr + 1; poolSize <=
         pytorch_qnnp_params.q8avgpool.mr + pytorch_qnnp_params.q8avgpool.qr;
         poolSize++) {
      AveragePoolingOperatorTester()
          .batchSize(1)
          .inputHeight(poolSize + 1)
          .inputWidth(3)
          .poolingHeight(poolSize)
          .poolingWidth(1)
          .channels(channels)
          .qmax(128)
          .testQ8();
      AveragePoolingOperatorTester()
          .batchSize(1)
          .inputHeight(2)
          .inputWidth(poolSize + 2)
          .poolingHeight(1)
          .poolingWidth(poolSize)
          .channels(channels)
          .qmax(128)
          .testQ8();
    }
  }
}

TEST(AVERAGE_POOLING_OP, unit_batch_few_channels_1xM_pool) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = 1; channels < pytorch_qnnp_params.q8avgpool.kr;
       channels++) {
    for (size_t poolSize = 2; poolSize <= 2 * pytorch_qnnp_params.q8avgpool.kr;
         poolSize++) {
      AveragePoolingOperatorTester()
          .batchSize(1)
          .inputHeight(2)
          .inputWidth(poolSize + 2)
          .poolingHeight(1)
          .poolingWidth(poolSize)
          .channels(channels)
          .testQ8();
    }
  }
}

TEST(AVERAGE_POOLING_OP, unit_batch_few_channels_1xM_pool_with_padding) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = 1; channels < pytorch_qnnp_params.q8avgpool.kr;
       channels++) {
    for (size_t poolSize = 3; poolSize <= pytorch_qnnp_params.q8avgpool.mr;
         poolSize++) {
      for (size_t paddingWidth = 0; paddingWidth <= 1; paddingWidth++) {
        AveragePoolingOperatorTester()
            .batchSize(1)
            .inputHeight(2)
            .inputWidth(poolSize + 2)
            .paddingWidth(paddingWidth)
            .poolingHeight(1)
            .poolingWidth(poolSize)
            .channels(channels)
            .testQ8();
      }
    }
  }
}

TEST(AVERAGE_POOLING_OP, unit_batch_few_channels_1xM_pool_with_stride) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = 1; channels < pytorch_qnnp_params.q8avgpool.kr;
       channels++) {
    for (size_t poolSize = 2; poolSize <= 2 * pytorch_qnnp_params.q8avgpool.kr;
         poolSize++) {
      AveragePoolingOperatorTester()
          .batchSize(1)
          .inputHeight(2)
          .inputWidth(poolSize + 4)
          .poolingHeight(1)
          .poolingWidth(poolSize)
          .strideWidth(2)
          .channels(channels)
          .testQ8();
    }
  }
}

TEST(AVERAGE_POOLING_OP, unit_batch_few_channels_Mx1_pool) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = 1; channels < pytorch_qnnp_params.q8avgpool.kr;
       channels++) {
    for (size_t poolSize = 2; poolSize <= 2 * pytorch_qnnp_params.q8avgpool.kr;
         poolSize++) {
      AveragePoolingOperatorTester()
          .batchSize(1)
          .inputHeight(poolSize + 1)
          .inputWidth(3)
          .poolingHeight(poolSize)
          .poolingWidth(1)
          .channels(channels)
          .testQ8();
    }
  }
}

TEST(AVERAGE_POOLING_OP, unit_batch_few_channels_Mx1_pool_with_padding) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = 1; channels < pytorch_qnnp_params.q8avgpool.kr;
       channels++) {
    for (size_t poolSize = 2; poolSize <= 2 * pytorch_qnnp_params.q8avgpool.kr;
         poolSize++) {
      for (size_t paddingHeight = 0; paddingHeight <= 1; paddingHeight++) {
        AveragePoolingOperatorTester()
            .batchSize(1)
            .inputHeight(poolSize + 1)
            .inputWidth(3)
            .paddingHeight(paddingHeight)
            .poolingHeight(poolSize)
            .poolingWidth(1)
            .channels(channels)
            .testQ8();
      }
    }
  }
}

TEST(AVERAGE_POOLING_OP, unit_batch_few_channels_Mx1_pool_with_stride) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = 1; channels < pytorch_qnnp_params.q8avgpool.kr;
       channels++) {
    for (size_t poolSize = 2; poolSize <= 2 * pytorch_qnnp_params.q8avgpool.kr;
         poolSize++) {
      AveragePoolingOperatorTester()
          .batchSize(1)
          .inputHeight(poolSize + 3)
          .inputWidth(3)
          .poolingHeight(poolSize)
          .poolingWidth(1)
          .strideHeight(2)
          .channels(channels)
          .testQ8();
    }
  }
}

TEST(AVERAGE_POOLING_OP, unit_batch_few_channels_with_input_stride) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = 1; channels < pytorch_qnnp_params.q8avgpool.kr;
       channels++) {
    for (size_t poolSize = 2; poolSize <= 2 * pytorch_qnnp_params.q8avgpool.kr;
         poolSize++) {
      AveragePoolingOperatorTester()
          .batchSize(1)
          .inputHeight(poolSize + 1)
          .inputWidth(3)
          .poolingHeight(poolSize)
          .poolingWidth(1)
          .channels(channels)
          .inputPixelStride(5 * pytorch_qnnp_params.q8avgpool.kr)
          .testQ8();
      AveragePoolingOperatorTester()
          .batchSize(1)
          .inputHeight(2)
          .inputWidth(poolSize + 2)
          .poolingHeight(1)
          .poolingWidth(poolSize)
          .channels(channels)
          .inputPixelStride(5 * pytorch_qnnp_params.q8avgpool.kr)
          .testQ8();
    }
  }
}

TEST(AVERAGE_POOLING_OP, unit_batch_few_channels_with_input_scale) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = 1; channels < pytorch_qnnp_params.q8avgpool.kr;
       channels++) {
    for (size_t poolSize = 2; poolSize <= 2 * pytorch_qnnp_params.q8avgpool.kr;
         poolSize++) {
      for (float inputScale = 0.01f; inputScale < 100.0f;
           inputScale *= 3.14159265f) {
        AveragePoolingOperatorTester()
            .batchSize(1)
            .inputHeight(poolSize + 1)
            .inputWidth(3)
            .poolingHeight(poolSize)
            .poolingWidth(1)
            .channels(channels)
            .inputScale(inputScale)
            .testQ8();
        AveragePoolingOperatorTester()
            .batchSize(1)
            .inputHeight(2)
            .inputWidth(poolSize + 2)
            .poolingHeight(1)
            .poolingWidth(poolSize)
            .channels(channels)
            .inputScale(inputScale)
            .testQ8();
      }
    }
  }
}

TEST(AVERAGE_POOLING_OP, unit_batch_few_channels_with_input_zero_point) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = 1; channels < pytorch_qnnp_params.q8avgpool.kr;
       channels++) {
    for (size_t poolSize = 2; poolSize <= 2 * pytorch_qnnp_params.q8avgpool.kr;
         poolSize++) {
      for (int32_t inputZeroPoint = 0; inputZeroPoint <= 255;
           inputZeroPoint += 51) {
        AveragePoolingOperatorTester()
            .batchSize(1)
            .inputHeight(poolSize + 1)
            .inputWidth(3)
            .poolingHeight(poolSize)
            .poolingWidth(1)
            .channels(channels)
            .inputZeroPoint(uint8_t(inputZeroPoint))
            .testQ8();
        AveragePoolingOperatorTester()
            .batchSize(1)
            .inputHeight(2)
            .inputWidth(poolSize + 2)
            .poolingHeight(1)
            .poolingWidth(poolSize)
            .channels(channels)
            .inputZeroPoint(uint8_t(inputZeroPoint))
            .testQ8();
      }
    }
  }
}

TEST(AVERAGE_POOLING_OP, unit_batch_few_channels_with_output_stride) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = 1; channels < pytorch_qnnp_params.q8avgpool.kr;
       channels++) {
    for (size_t poolSize = 2; poolSize <= 2 * pytorch_qnnp_params.q8avgpool.kr;
         poolSize++) {
      AveragePoolingOperatorTester()
          .batchSize(1)
          .inputHeight(poolSize + 1)
          .inputWidth(3)
          .poolingHeight(poolSize)
          .poolingWidth(1)
          .channels(channels)
          .outputPixelStride(5 * pytorch_qnnp_params.q8avgpool.kr)
          .testQ8();
      AveragePoolingOperatorTester()
          .batchSize(1)
          .inputHeight(2)
          .inputWidth(poolSize + 2)
          .poolingHeight(1)
          .poolingWidth(poolSize)
          .channels(channels)
          .outputPixelStride(5 * pytorch_qnnp_params.q8avgpool.kr)
          .testQ8();
    }
  }
}

TEST(AVERAGE_POOLING_OP, unit_batch_few_channels_with_output_scale) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = 1; channels < pytorch_qnnp_params.q8avgpool.kr;
       channels++) {
    for (size_t poolSize = 2; poolSize <= 2 * pytorch_qnnp_params.q8avgpool.kr;
         poolSize++) {
      for (float outputScale = 0.01f; outputScale < 100.0f;
           outputScale *= 3.14159265f) {
        AveragePoolingOperatorTester()
            .batchSize(1)
            .inputHeight(poolSize + 1)
            .inputWidth(3)
            .poolingHeight(poolSize)
            .poolingWidth(1)
            .channels(channels)
            .outputScale(outputScale)
            .testQ8();
        AveragePoolingOperatorTester()
            .batchSize(1)
            .inputHeight(2)
            .inputWidth(poolSize + 2)
            .poolingHeight(1)
            .poolingWidth(poolSize)
            .channels(channels)
            .outputScale(outputScale)
            .testQ8();
      }
    }
  }
}

TEST(AVERAGE_POOLING_OP, unit_batch_few_channels_with_output_zero_point) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = 1; channels < pytorch_qnnp_params.q8avgpool.kr;
       channels++) {
    for (size_t poolSize = 2; poolSize <= 2 * pytorch_qnnp_params.q8avgpool.kr;
         poolSize++) {
      for (int32_t outputZeroPoint = 0; outputZeroPoint <= 255;
           outputZeroPoint += 51) {
        AveragePoolingOperatorTester()
            .batchSize(1)
            .inputHeight(poolSize + 1)
            .inputWidth(3)
            .poolingHeight(poolSize)
            .poolingWidth(1)
            .channels(channels)
            .outputZeroPoint(uint8_t(outputZeroPoint))
            .testQ8();
        AveragePoolingOperatorTester()
            .batchSize(1)
            .inputHeight(2)
            .inputWidth(poolSize + 2)
            .poolingHeight(1)
            .poolingWidth(poolSize)
            .channels(channels)
            .outputZeroPoint(uint8_t(outputZeroPoint))
            .testQ8();
      }
    }
  }
}

TEST(AVERAGE_POOLING_OP, unit_batch_few_channels_with_qmin) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = 1; channels < pytorch_qnnp_params.q8avgpool.kr;
       channels++) {
    for (size_t poolSize = 2; poolSize <= 2 * pytorch_qnnp_params.q8avgpool.kr;
         poolSize++) {
      AveragePoolingOperatorTester()
          .batchSize(1)
          .inputHeight(poolSize + 1)
          .inputWidth(3)
          .poolingHeight(poolSize)
          .poolingWidth(1)
          .channels(channels)
          .qmin(128)
          .testQ8();
      AveragePoolingOperatorTester()
          .batchSize(1)
          .inputHeight(2)
          .inputWidth(poolSize + 1)
          .poolingHeight(1)
          .poolingWidth(poolSize)
          .channels(channels)
          .qmin(128)
          .testQ8();
    }
  }
}

TEST(AVERAGE_POOLING_OP, unit_batch_few_channels_with_qmax) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = 1; channels < pytorch_qnnp_params.q8avgpool.kr;
       channels++) {
    for (size_t poolSize = 2; poolSize <= 2 * pytorch_qnnp_params.q8avgpool.kr;
         poolSize++) {
      AveragePoolingOperatorTester()
          .batchSize(1)
          .inputHeight(poolSize + 1)
          .inputWidth(3)
          .poolingHeight(poolSize)
          .poolingWidth(1)
          .channels(channels)
          .qmax(128)
          .testQ8();
      AveragePoolingOperatorTester()
          .batchSize(1)
          .inputHeight(2)
          .inputWidth(poolSize + 1)
          .poolingHeight(1)
          .poolingWidth(poolSize)
          .channels(channels)
          .qmax(128)
          .testQ8();
    }
  }
}

TEST(AVERAGE_POOLING_OP, small_batch_many_channels_small_pool) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.q8avgpool.kr;
       channels <= 3 * pytorch_qnnp_params.q8avgpool.kr;
       channels++) {
    for (size_t poolSize = 2; poolSize <= pytorch_qnnp_params.q8avgpool.mr;
         poolSize++) {
      AveragePoolingOperatorTester()
          .batchSize(3)
          .inputHeight(poolSize + 1)
          .inputWidth(3)
          .poolingHeight(poolSize)
          .poolingWidth(1)
          .channels(channels)
          .testQ8();
      AveragePoolingOperatorTester()
          .batchSize(3)
          .inputHeight(2)
          .inputWidth(poolSize + 2)
          .poolingHeight(1)
          .poolingWidth(poolSize)
          .channels(channels)
          .testQ8();
    }
  }
}

TEST(
    AVERAGE_POOLING_OP,
    small_batch_many_channels_small_pool_with_input_stride) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.q8avgpool.kr;
       channels <= 3 * pytorch_qnnp_params.q8avgpool.kr;
       channels += 3) {
    for (size_t poolSize = 2; poolSize <= pytorch_qnnp_params.q8avgpool.mr;
         poolSize++) {
      AveragePoolingOperatorTester()
          .batchSize(3)
          .inputHeight(poolSize + 1)
          .inputWidth(3)
          .poolingHeight(poolSize)
          .poolingWidth(1)
          .channels(channels)
          .inputPixelStride(5 * pytorch_qnnp_params.q8avgpool.kr)
          .testQ8();
      AveragePoolingOperatorTester()
          .batchSize(3)
          .inputHeight(2)
          .inputWidth(poolSize + 1)
          .poolingHeight(1)
          .poolingWidth(poolSize)
          .channels(channels)
          .inputPixelStride(5 * pytorch_qnnp_params.q8avgpool.kr)
          .testQ8();
    }
  }
}

TEST(
    AVERAGE_POOLING_OP,
    small_batch_many_channels_small_pool_with_output_stride) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.q8avgpool.kr;
       channels <= 3 * pytorch_qnnp_params.q8avgpool.kr;
       channels += 3) {
    for (size_t poolSize = 2; poolSize <= pytorch_qnnp_params.q8avgpool.mr;
         poolSize++) {
      AveragePoolingOperatorTester()
          .batchSize(3)
          .inputHeight(poolSize + 1)
          .inputWidth(3)
          .poolingHeight(poolSize)
          .poolingWidth(1)
          .channels(channels)
          .outputPixelStride(5 * pytorch_qnnp_params.q8avgpool.kr)
          .testQ8();
      AveragePoolingOperatorTester()
          .batchSize(3)
          .inputHeight(2)
          .inputWidth(poolSize + 1)
          .poolingHeight(1)
          .poolingWidth(poolSize)
          .channels(channels)
          .outputPixelStride(5 * pytorch_qnnp_params.q8avgpool.kr)
          .testQ8();
    }
  }
}

TEST(AVERAGE_POOLING_OP, small_batch_many_channels_large_pool) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.q8avgpool.kr;
       channels <= 3 * pytorch_qnnp_params.q8avgpool.kr;
       channels++) {
    for (size_t poolSize = pytorch_qnnp_params.q8avgpool.mr + 1; poolSize <=
         pytorch_qnnp_params.q8avgpool.mr + pytorch_qnnp_params.q8avgpool.qr;
         poolSize++) {
      AveragePoolingOperatorTester()
          .batchSize(3)
          .inputHeight(poolSize + 1)
          .inputWidth(3)
          .poolingHeight(poolSize)
          .poolingWidth(1)
          .channels(channels)
          .testQ8();
      AveragePoolingOperatorTester()
          .batchSize(3)
          .inputHeight(2)
          .inputWidth(poolSize + 2)
          .poolingHeight(1)
          .poolingWidth(poolSize)
          .channels(channels)
          .testQ8();
    }
  }
}

TEST(
    AVERAGE_POOLING_OP,
    small_batch_many_channels_large_pool_with_input_stride) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.q8avgpool.kr;
       channels <= 3 * pytorch_qnnp_params.q8avgpool.kr;
       channels += 5) {
    for (size_t poolSize = pytorch_qnnp_params.q8avgpool.mr + 1; poolSize <=
         pytorch_qnnp_params.q8avgpool.mr + pytorch_qnnp_params.q8avgpool.qr;
         poolSize++) {
      AveragePoolingOperatorTester()
          .batchSize(3)
          .inputHeight(poolSize + 1)
          .inputWidth(3)
          .poolingHeight(poolSize)
          .poolingWidth(1)
          .channels(channels)
          .inputPixelStride(5 * pytorch_qnnp_params.q8avgpool.kr)
          .testQ8();
      AveragePoolingOperatorTester()
          .batchSize(3)
          .inputHeight(2)
          .inputWidth(poolSize + 1)
          .poolingHeight(1)
          .poolingWidth(poolSize)
          .channels(channels)
          .inputPixelStride(5 * pytorch_qnnp_params.q8avgpool.kr)
          .testQ8();
    }
  }
}

TEST(
    AVERAGE_POOLING_OP,
    small_batch_many_channels_large_pool_with_output_stride) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.q8avgpool.kr;
       channels <= 3 * pytorch_qnnp_params.q8avgpool.kr;
       channels += 5) {
    for (size_t poolSize = pytorch_qnnp_params.q8avgpool.mr + 1; poolSize <=
         pytorch_qnnp_params.q8avgpool.mr + pytorch_qnnp_params.q8avgpool.qr;
         poolSize++) {
      AveragePoolingOperatorTester()
          .batchSize(3)
          .inputHeight(poolSize + 1)
          .inputWidth(3)
          .poolingHeight(poolSize)
          .poolingWidth(1)
          .channels(channels)
          .outputPixelStride(5 * pytorch_qnnp_params.q8avgpool.kr)
          .testQ8();
      AveragePoolingOperatorTester()
          .batchSize(3)
          .inputHeight(2)
          .inputWidth(poolSize + 1)
          .poolingHeight(1)
          .poolingWidth(poolSize)
          .channels(channels)
          .outputPixelStride(5 * pytorch_qnnp_params.q8avgpool.kr)
          .testQ8();
    }
  }
}

TEST(AVERAGE_POOLING_OP, small_batch_few_channels) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = 1; channels < pytorch_qnnp_params.q8avgpool.kr;
       channels++) {
    for (size_t poolSize = 2; poolSize <= 2 * pytorch_qnnp_params.q8avgpool.kr;
         poolSize++) {
      AveragePoolingOperatorTester()
          .batchSize(3)
          .inputHeight(poolSize + 1)
          .inputWidth(3)
          .poolingHeight(poolSize)
          .poolingWidth(1)
          .channels(channels)
          .testQ8();
      AveragePoolingOperatorTester()
          .batchSize(3)
          .inputHeight(2)
          .inputWidth(poolSize + 2)
          .poolingHeight(1)
          .poolingWidth(poolSize)
          .channels(channels)
          .testQ8();
    }
  }
}

TEST(AVERAGE_POOLING_OP, small_batch_few_channels_with_input_stride) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = 1; channels < pytorch_qnnp_params.q8avgpool.kr;
       channels++) {
    for (size_t poolSize = 2; poolSize <= 2 * pytorch_qnnp_params.q8avgpool.kr;
         poolSize += 3) {
      AveragePoolingOperatorTester()
          .batchSize(3)
          .inputHeight(poolSize + 1)
          .inputWidth(3)
          .poolingHeight(poolSize)
          .poolingWidth(1)
          .channels(channels)
          .inputPixelStride(5 * pytorch_qnnp_params.q8avgpool.kr)
          .testQ8();
      AveragePoolingOperatorTester()
          .batchSize(3)
          .inputHeight(2)
          .inputWidth(poolSize + 2)
          .poolingHeight(1)
          .poolingWidth(poolSize)
          .channels(channels)
          .inputPixelStride(5 * pytorch_qnnp_params.q8avgpool.kr)
          .testQ8();
    }
  }
}

TEST(AVERAGE_POOLING_OP, small_batch_few_channels_with_output_stride) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = 1; channels < pytorch_qnnp_params.q8avgpool.kr;
       channels++) {
    for (size_t poolSize = 2; poolSize <= 2 * pytorch_qnnp_params.q8avgpool.kr;
         poolSize += 3) {
      AveragePoolingOperatorTester()
          .batchSize(3)
          .inputHeight(poolSize + 1)
          .inputWidth(3)
          .poolingHeight(poolSize)
          .poolingWidth(1)
          .channels(channels)
          .outputPixelStride(5 * pytorch_qnnp_params.q8avgpool.kr)
          .testQ8();
      AveragePoolingOperatorTester()
          .batchSize(3)
          .inputHeight(2)
          .inputWidth(poolSize + 2)
          .poolingHeight(1)
          .poolingWidth(poolSize)
          .channels(channels)
          .outputPixelStride(5 * pytorch_qnnp_params.q8avgpool.kr)
          .testQ8();
    }
  }
}

TEST(AVERAGE_POOLING_OP, setup_increasing_batch) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  AveragePoolingOperatorTester()
      .batchSize(3)
      .nextBatchSize(5)
      .inputHeight(8)
      .inputWidth(8)
      .poolingHeight(5)
      .poolingWidth(3)
      .channels(24)
      .testSetupQ8();
}

TEST(AVERAGE_POOLING_OP, setup_decreasing_batch) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  AveragePoolingOperatorTester()
      .batchSize(5)
      .nextBatchSize(3)
      .inputHeight(8)
      .inputWidth(8)
      .poolingHeight(5)
      .poolingWidth(3)
      .channels(24)
      .testSetupQ8();
}

TEST(AVERAGE_POOLING_OP, setup_changing_height) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  AveragePoolingOperatorTester()
      .batchSize(3)
      .inputHeight(8)
      .inputWidth(8)
      .nextInputHeight(9)
      .poolingHeight(5)
      .poolingWidth(3)
      .channels(24)
      .testSetupQ8();
  AveragePoolingOperatorTester()
      .batchSize(3)
      .inputHeight(8)
      .inputWidth(8)
      .nextInputHeight(7)
      .poolingHeight(5)
      .poolingWidth(3)
      .channels(24)
      .testSetupQ8();
}

TEST(AVERAGE_POOLING_OP, setup_changing_width) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  AveragePoolingOperatorTester()
      .batchSize(3)
      .inputHeight(8)
      .inputWidth(8)
      .nextInputWidth(9)
      .poolingHeight(5)
      .poolingWidth(3)
      .channels(24)
      .testSetupQ8();
  AveragePoolingOperatorTester()
      .batchSize(3)
      .inputHeight(8)
      .inputWidth(8)
      .nextInputWidth(7)
      .poolingHeight(5)
      .poolingWidth(3)
      .channels(24)
      .testSetupQ8();
}

TEST(AVERAGE_POOLING_OP, setup_swap_height_and_width) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  AveragePoolingOperatorTester()
      .batchSize(3)
      .inputHeight(9)
      .inputWidth(8)
      .nextInputHeight(8)
      .nextInputWidth(9)
      .poolingHeight(5)
      .poolingWidth(3)
      .channels(24)
      .testSetupQ8();
}
