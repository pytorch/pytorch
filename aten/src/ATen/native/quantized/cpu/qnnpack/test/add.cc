/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "add-operator-tester.h"

TEST(ADD_OP, zero_batch) {
  AddOperatorTester().batchSize(0).channels(2).iterations(1).testQ8();
}

TEST(ADD_OP, unit_batch) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    AddOperatorTester().batchSize(1).channels(channels).iterations(3).testQ8();
  }
}

TEST(ADD_OP, unit_batch_with_qmin) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    AddOperatorTester()
        .batchSize(1)
        .channels(channels)
        .qmin(128)
        .iterations(3)
        .testQ8();
  }
}

TEST(ADD_OP, unit_batch_with_qmax) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    AddOperatorTester()
        .batchSize(1)
        .channels(channels)
        .qmax(128)
        .iterations(3)
        .testQ8();
  }
}

TEST(ADD_OP, unit_batch_with_a_scale) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (float aScale = 1.0e-2f; aScale < 1.0e+2f; aScale *= 10.0f) {
      AddOperatorTester()
          .batchSize(1)
          .channels(channels)
          .aScale(aScale)
          .iterations(1)
          .testQ8();
    }
  }
}

TEST(ADD_OP, unit_batch_with_b_scale) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (float bScale = 1.0e-2f; bScale < 1.0e+2f; bScale *= 10.0f) {
      AddOperatorTester()
          .batchSize(1)
          .channels(channels)
          .bScale(bScale)
          .iterations(1)
          .testQ8();
    }
  }
}

TEST(ADD_OP, unit_batch_with_y_scale) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (float yScale = 1.0e-2f; yScale < 1.0e+2f; yScale *= 10.0f) {
      AddOperatorTester()
          .batchSize(1)
          .channels(channels)
          .yScale(yScale)
          .iterations(1)
          .testQ8();
    }
  }
}

TEST(ADD_OP, unit_batch_with_a_zero_point) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (int32_t aZeroPoint = 0; aZeroPoint <= 255; aZeroPoint += 51) {
      AddOperatorTester()
          .batchSize(1)
          .channels(channels)
          .aZeroPoint(uint8_t(aZeroPoint))
          .iterations(1)
          .testQ8();
    }
  }
}

TEST(ADD_OP, unit_batch_with_b_zero_point) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (int32_t bZeroPoint = 0; bZeroPoint <= 255; bZeroPoint += 51) {
      AddOperatorTester()
          .batchSize(1)
          .channels(channels)
          .bZeroPoint(uint8_t(bZeroPoint))
          .iterations(1)
          .testQ8();
    }
  }
}

TEST(ADD_OP, unit_batch_with_y_zero_point) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (int32_t yZeroPoint = 0; yZeroPoint <= 255; yZeroPoint += 51) {
      AddOperatorTester()
          .batchSize(1)
          .channels(channels)
          .yZeroPoint(uint8_t(yZeroPoint))
          .iterations(1)
          .testQ8();
    }
  }
}

TEST(ADD_OP, small_batch) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    AddOperatorTester().batchSize(3).channels(channels).iterations(3).testQ8();
  }
}

TEST(ADD_OP, small_batch_with_a_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    AddOperatorTester()
        .batchSize(3)
        .channels(channels)
        .aStride(129)
        .iterations(3)
        .testQ8();
  }
}

TEST(ADD_OP, small_batch_with_b_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    AddOperatorTester()
        .batchSize(3)
        .channels(channels)
        .bStride(123)
        .iterations(3)
        .testQ8();
  }
}

TEST(ADD_OP, small_batch_with_y_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    AddOperatorTester()
        .batchSize(3)
        .channels(channels)
        .yStride(117)
        .iterations(3)
        .testQ8();
  }
}

TEST(ADD_OP, small_batch_with_qmin) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    AddOperatorTester()
        .batchSize(3)
        .channels(channels)
        .qmin(128)
        .iterations(3)
        .testQ8();
  }
}

TEST(ADD_OP, small_batch_with_qmax) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    AddOperatorTester()
        .batchSize(3)
        .channels(channels)
        .qmax(128)
        .iterations(3)
        .testQ8();
  }
}

TEST(ADD_OP, small_batch_with_a_scale) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (float aScale = 1.0e-2f; aScale < 1.0e+2f; aScale *= 10.0f) {
      AddOperatorTester()
          .batchSize(3)
          .channels(channels)
          .aScale(aScale)
          .iterations(1)
          .testQ8();
    }
  }
}

TEST(ADD_OP, small_batch_with_b_scale) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (float bScale = 1.0e-2f; bScale < 1.0e+2f; bScale *= 10.0f) {
      AddOperatorTester()
          .batchSize(3)
          .channels(channels)
          .bScale(bScale)
          .iterations(1)
          .testQ8();
    }
  }
}

TEST(ADD_OP, small_batch_with_y_scale) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (float yScale = 1.0e-2f; yScale < 1.0e+2f; yScale *= 10.0f) {
      AddOperatorTester()
          .batchSize(3)
          .channels(channels)
          .yScale(yScale)
          .iterations(1)
          .testQ8();
    }
  }
}

TEST(ADD_OP, small_batch_with_a_zero_point) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (int32_t aZeroPoint = 0; aZeroPoint <= 255; aZeroPoint += 51) {
      AddOperatorTester()
          .batchSize(3)
          .channels(channels)
          .aZeroPoint(uint8_t(aZeroPoint))
          .iterations(1)
          .testQ8();
    }
  }
}

TEST(ADD_OP, small_batch_with_b_zero_point) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (int32_t bZeroPoint = 0; bZeroPoint <= 255; bZeroPoint += 51) {
      AddOperatorTester()
          .batchSize(3)
          .channels(channels)
          .bZeroPoint(uint8_t(bZeroPoint))
          .iterations(1)
          .testQ8();
    }
  }
}

TEST(ADD_OP, small_batch_with_y_zero_point) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (int32_t yZeroPoint = 0; yZeroPoint <= 255; yZeroPoint += 51) {
      AddOperatorTester()
          .batchSize(3)
          .channels(channels)
          .yZeroPoint(uint8_t(yZeroPoint))
          .iterations(1)
          .testQ8();
    }
  }
}

TEST(ADD_OP, strided_batch) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    AddOperatorTester()
        .batchSize(3)
        .channels(channels)
        .aStride(129)
        .bStride(123)
        .yStride(117)
        .iterations(3)
        .testQ8();
  }
}

TEST(ADD_OP, strided_batch_with_qmin) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    AddOperatorTester()
        .batchSize(3)
        .channels(channels)
        .aStride(129)
        .bStride(123)
        .yStride(117)
        .qmin(128)
        .iterations(3)
        .testQ8();
  }
}

TEST(ADD_OP, strided_batch_with_qmax) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    AddOperatorTester()
        .batchSize(3)
        .channels(channels)
        .aStride(129)
        .bStride(123)
        .yStride(117)
        .qmax(128)
        .iterations(3)
        .testQ8();
  }
}

TEST(ADD_OP, strided_batch_with_a_scale) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (float aScale = 1.0e-2f; aScale < 1.0e+2f; aScale *= 10.0f) {
      AddOperatorTester()
          .batchSize(3)
          .channels(channels)
          .aStride(129)
          .bStride(123)
          .yStride(117)
          .aScale(aScale)
          .iterations(1)
          .testQ8();
    }
  }
}

TEST(ADD_OP, strided_batch_with_b_scale) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (float bScale = 1.0e-2f; bScale < 1.0e+2f; bScale *= 10.0f) {
      AddOperatorTester()
          .batchSize(3)
          .channels(channels)
          .aStride(129)
          .bStride(123)
          .yStride(117)
          .bScale(bScale)
          .iterations(1)
          .testQ8();
    }
  }
}

TEST(ADD_OP, strided_batch_with_y_scale) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (float yScale = 1.0e-2f; yScale < 1.0e+2f; yScale *= 10.0f) {
      AddOperatorTester()
          .batchSize(3)
          .channels(channels)
          .aStride(129)
          .bStride(123)
          .yStride(117)
          .yScale(yScale)
          .iterations(1)
          .testQ8();
    }
  }
}

TEST(ADD_OP, strided_batch_with_a_zero_point) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (int32_t aZeroPoint = 0; aZeroPoint <= 255; aZeroPoint += 51) {
      AddOperatorTester()
          .batchSize(3)
          .channels(channels)
          .aStride(129)
          .bStride(123)
          .yStride(117)
          .aZeroPoint(uint8_t(aZeroPoint))
          .iterations(1)
          .testQ8();
    }
  }
}

TEST(ADD_OP, strided_batch_with_b_zero_point) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (int32_t bZeroPoint = 0; bZeroPoint <= 255; bZeroPoint += 51) {
      AddOperatorTester()
          .batchSize(3)
          .channels(channels)
          .aStride(129)
          .bStride(123)
          .yStride(117)
          .bZeroPoint(uint8_t(bZeroPoint))
          .iterations(1)
          .testQ8();
    }
  }
}

TEST(ADD_OP, strided_batch_with_y_zero_point) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (int32_t yZeroPoint = 0; yZeroPoint <= 255; yZeroPoint += 51) {
      AddOperatorTester()
          .batchSize(3)
          .channels(channels)
          .aStride(129)
          .bStride(123)
          .yStride(117)
          .yZeroPoint(uint8_t(yZeroPoint))
          .iterations(1)
          .testQ8();
    }
  }
}
