/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "global-average-pooling-operator-tester.h"

#include <qnnpack/params.h>

TEST(GLOBAL_AVERAGE_POOLING_OP, zero_batch) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  GlobalAveragePoolingOperatorTester()
      .batchSize(0)
      .width(1)
      .channels(8)
      .testQ8();
}

TEST(GLOBAL_AVERAGE_POOLING_OP, unit_batch_many_channels_small_width) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.q8gavgpool.nr;
       channels <= 3 * pytorch_qnnp_params.q8gavgpool.nr;
       channels++) {
    for (size_t width = 1; width <= pytorch_qnnp_params.q8gavgpool.mr;
         width++) {
      GlobalAveragePoolingOperatorTester()
          .batchSize(1)
          .width(width)
          .channels(channels)
          .testQ8();
    }
  }
}

TEST(
    GLOBAL_AVERAGE_POOLING_OP,
    unit_batch_many_channels_small_width_with_input_stride) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.q8gavgpool.nr;
       channels <= 3 * pytorch_qnnp_params.q8gavgpool.nr;
       channels++) {
    for (size_t width = 1; width <= pytorch_qnnp_params.q8gavgpool.mr;
         width++) {
      GlobalAveragePoolingOperatorTester()
          .batchSize(1)
          .width(width)
          .channels(channels)
          .inputStride(5 * pytorch_qnnp_params.q8gavgpool.nr)
          .testQ8();
    }
  }
}

TEST(
    GLOBAL_AVERAGE_POOLING_OP,
    unit_batch_many_channels_small_width_with_input_scale) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.q8gavgpool.nr;
       channels <= 3 * pytorch_qnnp_params.q8gavgpool.nr;
       channels++) {
    for (size_t width = 1; width <= pytorch_qnnp_params.q8gavgpool.mr;
         width++) {
      for (float inputScale = 0.01f; inputScale < 100.0f;
           inputScale *= 3.14159265f) {
        GlobalAveragePoolingOperatorTester()
            .batchSize(1)
            .width(width)
            .channels(channels)
            .inputScale(inputScale)
            .testQ8();
      }
    }
  }
}

TEST(
    GLOBAL_AVERAGE_POOLING_OP,
    unit_batch_many_channels_small_width_with_input_zero_point) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.q8gavgpool.nr;
       channels <= 3 * pytorch_qnnp_params.q8gavgpool.nr;
       channels++) {
    for (size_t width = 1; width <= pytorch_qnnp_params.q8gavgpool.mr;
         width++) {
      for (int32_t inputZeroPoint = 0; inputZeroPoint <= 255;
           inputZeroPoint += 51) {
        GlobalAveragePoolingOperatorTester()
            .batchSize(1)
            .width(width)
            .channels(channels)
            .inputZeroPoint(uint8_t(inputZeroPoint))
            .testQ8();
      }
    }
  }
}

TEST(
    GLOBAL_AVERAGE_POOLING_OP,
    unit_batch_many_channels_small_width_with_output_scale) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.q8gavgpool.nr;
       channels <= 3 * pytorch_qnnp_params.q8gavgpool.nr;
       channels++) {
    for (size_t width = 1; width <= pytorch_qnnp_params.q8gavgpool.mr;
         width++) {
      for (float outputScale = 0.01f; outputScale < 100.0f;
           outputScale *= 3.14159265f) {
        GlobalAveragePoolingOperatorTester()
            .batchSize(1)
            .width(width)
            .channels(channels)
            .outputScale(outputScale)
            .testQ8();
      }
    }
  }
}

TEST(
    GLOBAL_AVERAGE_POOLING_OP,
    unit_batch_many_channels_small_width_with_output_zero_point) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.q8gavgpool.nr;
       channels <= 3 * pytorch_qnnp_params.q8gavgpool.nr;
       channels++) {
    for (size_t width = 1; width <= pytorch_qnnp_params.q8gavgpool.mr;
         width++) {
      for (int32_t outputZeroPoint = 0; outputZeroPoint <= 255;
           outputZeroPoint += 51) {
        GlobalAveragePoolingOperatorTester()
            .batchSize(1)
            .width(width)
            .channels(channels)
            .outputZeroPoint(uint8_t(outputZeroPoint))
            .testQ8();
      }
    }
  }
}

TEST(
    GLOBAL_AVERAGE_POOLING_OP,
    unit_batch_many_channels_small_width_with_output_min) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.q8gavgpool.nr;
       channels <= 3 * pytorch_qnnp_params.q8gavgpool.nr;
       channels++) {
    for (size_t width = 1; width <= pytorch_qnnp_params.q8gavgpool.mr;
         width++) {
      GlobalAveragePoolingOperatorTester()
          .batchSize(1)
          .width(width)
          .channels(channels)
          .outputMin(128)
          .testQ8();
    }
  }
}

TEST(
    GLOBAL_AVERAGE_POOLING_OP,
    unit_batch_many_channels_small_width_with_output_max) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.q8gavgpool.nr;
       channels <= 3 * pytorch_qnnp_params.q8gavgpool.nr;
       channels++) {
    for (size_t width = 1; width <= pytorch_qnnp_params.q8gavgpool.mr;
         width++) {
      GlobalAveragePoolingOperatorTester()
          .batchSize(1)
          .width(width)
          .channels(channels)
          .outputMax(128)
          .testQ8();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_OP, unit_batch_many_channels_large_width) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.q8gavgpool.nr;
       channels <= 3 * pytorch_qnnp_params.q8gavgpool.nr;
       channels++) {
    for (size_t width = pytorch_qnnp_params.q8gavgpool.mr;
         width <= 4 * pytorch_qnnp_params.q8gavgpool.mr;
         width++) {
      GlobalAveragePoolingOperatorTester()
          .batchSize(1)
          .width(width)
          .channels(channels)
          .testQ8();
    }
  }
}

TEST(
    GLOBAL_AVERAGE_POOLING_OP,
    unit_batch_many_channels_large_width_with_input_stride) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.q8gavgpool.nr;
       channels <= 3 * pytorch_qnnp_params.q8gavgpool.nr;
       channels++) {
    for (size_t width = pytorch_qnnp_params.q8gavgpool.mr;
         width <= 4 * pytorch_qnnp_params.q8gavgpool.mr;
         width++) {
      GlobalAveragePoolingOperatorTester()
          .batchSize(1)
          .width(width)
          .channels(channels)
          .inputStride(5 * pytorch_qnnp_params.q8gavgpool.nr)
          .testQ8();
    }
  }
}

TEST(
    GLOBAL_AVERAGE_POOLING_OP,
    unit_batch_many_channels_large_width_with_input_scale) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.q8gavgpool.nr;
       channels <= 3 * pytorch_qnnp_params.q8gavgpool.nr;
       channels++) {
    for (size_t width = pytorch_qnnp_params.q8gavgpool.mr;
         width <= 4 * pytorch_qnnp_params.q8gavgpool.mr;
         width++) {
      for (float inputScale = 0.01f; inputScale < 100.0f;
           inputScale *= 3.14159265f) {
        GlobalAveragePoolingOperatorTester()
            .batchSize(1)
            .width(width)
            .channels(channels)
            .inputScale(inputScale)
            .testQ8();
      }
    }
  }
}

TEST(
    GLOBAL_AVERAGE_POOLING_OP,
    unit_batch_many_channels_large_width_with_input_zero_point) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.q8gavgpool.nr;
       channels <= 3 * pytorch_qnnp_params.q8gavgpool.nr;
       channels++) {
    for (size_t width = pytorch_qnnp_params.q8gavgpool.mr;
         width <= 4 * pytorch_qnnp_params.q8gavgpool.mr;
         width++) {
      for (int32_t inputZeroPoint = 0; inputZeroPoint <= 255;
           inputZeroPoint += 51) {
        GlobalAveragePoolingOperatorTester()
            .batchSize(1)
            .width(width)
            .channels(channels)
            .inputZeroPoint(uint8_t(inputZeroPoint))
            .testQ8();
      }
    }
  }
}

TEST(
    GLOBAL_AVERAGE_POOLING_OP,
    unit_batch_many_channels_large_width_with_output_scale) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.q8gavgpool.nr;
       channels <= 3 * pytorch_qnnp_params.q8gavgpool.nr;
       channels++) {
    for (size_t width = pytorch_qnnp_params.q8gavgpool.mr;
         width <= 4 * pytorch_qnnp_params.q8gavgpool.mr;
         width++) {
      for (float outputScale = 0.01f; outputScale < 100.0f;
           outputScale *= 3.14159265f) {
        GlobalAveragePoolingOperatorTester()
            .batchSize(1)
            .width(width)
            .channels(channels)
            .outputScale(outputScale)
            .testQ8();
      }
    }
  }
}

TEST(
    GLOBAL_AVERAGE_POOLING_OP,
    unit_batch_many_channels_large_width_with_output_zero_point) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.q8gavgpool.nr;
       channels <= 3 * pytorch_qnnp_params.q8gavgpool.nr;
       channels++) {
    for (size_t width = pytorch_qnnp_params.q8gavgpool.mr;
         width <= 4 * pytorch_qnnp_params.q8gavgpool.mr;
         width++) {
      for (int32_t outputZeroPoint = 0; outputZeroPoint <= 255;
           outputZeroPoint += 51) {
        GlobalAveragePoolingOperatorTester()
            .batchSize(1)
            .width(width)
            .channels(channels)
            .outputZeroPoint(uint8_t(outputZeroPoint))
            .testQ8();
      }
    }
  }
}

TEST(
    GLOBAL_AVERAGE_POOLING_OP,
    unit_batch_many_channels_large_width_with_output_min) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.q8gavgpool.nr;
       channels <= 3 * pytorch_qnnp_params.q8gavgpool.nr;
       channels++) {
    for (size_t width = pytorch_qnnp_params.q8gavgpool.mr;
         width <= 4 * pytorch_qnnp_params.q8gavgpool.mr;
         width++) {
      GlobalAveragePoolingOperatorTester()
          .batchSize(1)
          .width(width)
          .channels(channels)
          .outputMin(128)
          .testQ8();
    }
  }
}

TEST(
    GLOBAL_AVERAGE_POOLING_OP,
    unit_batch_many_channels_large_width_with_output_max) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.q8gavgpool.nr;
       channels <= 3 * pytorch_qnnp_params.q8gavgpool.nr;
       channels++) {
    for (size_t width = pytorch_qnnp_params.q8gavgpool.mr;
         width <= 4 * pytorch_qnnp_params.q8gavgpool.mr;
         width++) {
      GlobalAveragePoolingOperatorTester()
          .batchSize(1)
          .width(width)
          .channels(channels)
          .outputMax(128)
          .testQ8();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_OP, unit_batch_few_channels) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = 1; channels < pytorch_qnnp_params.q8gavgpool.nr;
       channels++) {
    for (size_t width = 1; width <= 2 * pytorch_qnnp_params.q8gavgpool.nr;
         width++) {
      GlobalAveragePoolingOperatorTester()
          .batchSize(1)
          .width(width)
          .channels(channels)
          .testQ8();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_OP, unit_batch_few_channels_with_input_stride) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = 1; channels < pytorch_qnnp_params.q8gavgpool.nr;
       channels++) {
    for (size_t width = 1; width <= 2 * pytorch_qnnp_params.q8gavgpool.nr;
         width++) {
      GlobalAveragePoolingOperatorTester()
          .batchSize(1)
          .width(width)
          .channels(channels)
          .inputStride(5 * pytorch_qnnp_params.q8gavgpool.nr)
          .testQ8();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_OP, unit_batch_few_channels_with_input_scale) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = 1; channels < pytorch_qnnp_params.q8gavgpool.nr;
       channels++) {
    for (size_t width = 1; width <= 2 * pytorch_qnnp_params.q8gavgpool.nr;
         width++) {
      for (float inputScale = 0.01f; inputScale < 100.0f;
           inputScale *= 3.14159265f) {
        GlobalAveragePoolingOperatorTester()
            .batchSize(1)
            .width(width)
            .channels(channels)
            .inputScale(inputScale)
            .testQ8();
      }
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_OP, unit_batch_few_channels_with_input_zero_point) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = 1; channels < pytorch_qnnp_params.q8gavgpool.nr;
       channels++) {
    for (size_t width = 1; width <= 2 * pytorch_qnnp_params.q8gavgpool.nr;
         width++) {
      for (int32_t inputZeroPoint = 0; inputZeroPoint <= 255;
           inputZeroPoint += 51) {
        GlobalAveragePoolingOperatorTester()
            .batchSize(1)
            .width(width)
            .channels(channels)
            .inputZeroPoint(uint8_t(inputZeroPoint))
            .testQ8();
      }
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_OP, unit_batch_few_channels_with_output_scale) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = 1; channels < pytorch_qnnp_params.q8gavgpool.nr;
       channels++) {
    for (size_t width = 1; width <= 2 * pytorch_qnnp_params.q8gavgpool.nr;
         width++) {
      for (float outputScale = 0.01f; outputScale < 100.0f;
           outputScale *= 3.14159265f) {
        GlobalAveragePoolingOperatorTester()
            .batchSize(1)
            .width(width)
            .channels(channels)
            .outputScale(outputScale)
            .testQ8();
      }
    }
  }
}

TEST(
    GLOBAL_AVERAGE_POOLING_OP,
    unit_batch_few_channels_with_output_zero_point) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = 1; channels < pytorch_qnnp_params.q8gavgpool.nr;
       channels++) {
    for (size_t width = 1; width <= 2 * pytorch_qnnp_params.q8gavgpool.nr;
         width++) {
      for (int32_t outputZeroPoint = 0; outputZeroPoint <= 255;
           outputZeroPoint += 51) {
        GlobalAveragePoolingOperatorTester()
            .batchSize(1)
            .width(width)
            .channels(channels)
            .outputZeroPoint(uint8_t(outputZeroPoint))
            .testQ8();
      }
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_OP, unit_batch_few_channels_with_output_min) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = 1; channels < pytorch_qnnp_params.q8gavgpool.nr;
       channels++) {
    for (size_t width = 1; width <= 2 * pytorch_qnnp_params.q8gavgpool.nr;
         width++) {
      GlobalAveragePoolingOperatorTester()
          .batchSize(1)
          .width(width)
          .channels(channels)
          .outputMin(128)
          .testQ8();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_OP, unit_batch_few_channels_with_output_max) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = 1; channels < pytorch_qnnp_params.q8gavgpool.nr;
       channels++) {
    for (size_t width = 1; width <= 2 * pytorch_qnnp_params.q8gavgpool.nr;
         width++) {
      GlobalAveragePoolingOperatorTester()
          .batchSize(1)
          .width(width)
          .channels(channels)
          .outputMax(128)
          .testQ8();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_OP, small_batch_many_channels_small_width) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.q8gavgpool.nr;
       channels <= 3 * pytorch_qnnp_params.q8gavgpool.nr;
       channels++) {
    for (size_t width = 1; width <= pytorch_qnnp_params.q8gavgpool.mr;
         width++) {
      GlobalAveragePoolingOperatorTester()
          .batchSize(3)
          .width(width)
          .channels(channels)
          .testQ8();
    }
  }
}

TEST(
    GLOBAL_AVERAGE_POOLING_OP,
    small_batch_many_channels_small_width_with_input_stride) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.q8gavgpool.nr;
       channels <= 3 * pytorch_qnnp_params.q8gavgpool.nr;
       channels++) {
    for (size_t width = 1; width <= pytorch_qnnp_params.q8gavgpool.mr;
         width++) {
      GlobalAveragePoolingOperatorTester()
          .batchSize(3)
          .width(width)
          .channels(channels)
          .inputStride(5 * pytorch_qnnp_params.q8gavgpool.nr)
          .testQ8();
    }
  }
}

TEST(
    GLOBAL_AVERAGE_POOLING_OP,
    small_batch_many_channels_small_width_with_output_stride) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.q8gavgpool.nr;
       channels <= 3 * pytorch_qnnp_params.q8gavgpool.nr;
       channels++) {
    for (size_t width = 1; width <= pytorch_qnnp_params.q8gavgpool.mr;
         width++) {
      GlobalAveragePoolingOperatorTester()
          .batchSize(3)
          .width(width)
          .channels(channels)
          .outputStride(5 * pytorch_qnnp_params.q8gavgpool.nr)
          .testQ8();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_OP, small_batch_many_channels_large_width) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.q8gavgpool.nr;
       channels <= 3 * pytorch_qnnp_params.q8gavgpool.nr;
       channels++) {
    for (size_t width = pytorch_qnnp_params.q8gavgpool.mr;
         width <= 4 * pytorch_qnnp_params.q8gavgpool.mr;
         width++) {
      GlobalAveragePoolingOperatorTester()
          .batchSize(3)
          .width(width)
          .channels(channels)
          .testQ8();
    }
  }
}

TEST(
    GLOBAL_AVERAGE_POOLING_OP,
    small_batch_many_channels_large_width_with_input_stride) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.q8gavgpool.nr;
       channels <= 3 * pytorch_qnnp_params.q8gavgpool.nr;
       channels++) {
    for (size_t width = pytorch_qnnp_params.q8gavgpool.mr;
         width <= 4 * pytorch_qnnp_params.q8gavgpool.mr;
         width++) {
      GlobalAveragePoolingOperatorTester()
          .batchSize(3)
          .width(width)
          .channels(channels)
          .inputStride(5 * pytorch_qnnp_params.q8gavgpool.nr)
          .testQ8();
    }
  }
}

TEST(
    GLOBAL_AVERAGE_POOLING_OP,
    small_batch_many_channels_large_width_with_output_stride) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = pytorch_qnnp_params.q8gavgpool.nr;
       channels <= 3 * pytorch_qnnp_params.q8gavgpool.nr;
       channels++) {
    for (size_t width = pytorch_qnnp_params.q8gavgpool.mr;
         width <= 4 * pytorch_qnnp_params.q8gavgpool.mr;
         width++) {
      GlobalAveragePoolingOperatorTester()
          .batchSize(3)
          .width(width)
          .channels(channels)
          .outputStride(5 * pytorch_qnnp_params.q8gavgpool.nr)
          .testQ8();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_OP, small_batch_few_channels) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = 1; channels < pytorch_qnnp_params.q8gavgpool.nr;
       channels++) {
    for (size_t width = 1; width <= 2 * pytorch_qnnp_params.q8gavgpool.nr;
         width++) {
      GlobalAveragePoolingOperatorTester()
          .batchSize(3)
          .width(width)
          .channels(channels)
          .testQ8();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_OP, small_batch_few_channels_with_input_stride) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = 1; channels < pytorch_qnnp_params.q8gavgpool.nr;
       channels++) {
    for (size_t width = 1; width <= 2 * pytorch_qnnp_params.q8gavgpool.nr;
         width++) {
      GlobalAveragePoolingOperatorTester()
          .batchSize(3)
          .width(width)
          .channels(channels)
          .inputStride(5 * pytorch_qnnp_params.q8gavgpool.nr)
          .testQ8();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_OP, small_batch_few_channels_with_output_stride) {
  ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
  for (size_t channels = 1; channels < pytorch_qnnp_params.q8gavgpool.nr;
       channels++) {
    for (size_t width = 1; width <= 2 * pytorch_qnnp_params.q8gavgpool.nr;
         width++) {
      GlobalAveragePoolingOperatorTester()
          .batchSize(3)
          .width(width)
          .channels(channels)
          .outputStride(5 * pytorch_qnnp_params.q8gavgpool.nr)
          .testQ8();
    }
  }
}
