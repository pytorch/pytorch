/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "channel-shuffle-operator-tester.h"

TEST(CHANNEL_SHUFFLE_OP, zero_batch) {
  ChannelShuffleOperatorTester()
      .batchSize(0)
      .groups(2)
      .groupChannels(4)
      .iterations(1)
      .testX8();
}

TEST(CHANNEL_SHUFFLE_OP, two_groups_unit_batch) {
  for (size_t groupChannels = 1; groupChannels < 100; groupChannels += 15) {
    ChannelShuffleOperatorTester()
        .batchSize(1)
        .groups(2)
        .groupChannels(groupChannels)
        .iterations(3)
        .testX8();
  }
}

TEST(CHANNEL_SHUFFLE_OP, three_groups_unit_batch) {
  for (size_t groupChannels = 1; groupChannels < 100; groupChannels += 15) {
    ChannelShuffleOperatorTester()
        .batchSize(1)
        .groups(3)
        .groupChannels(groupChannels)
        .iterations(3)
        .testX8();
  }
}

TEST(CHANNEL_SHUFFLE_OP, four_groups_unit_batch) {
  for (size_t groupChannels = 1; groupChannels < 100; groupChannels += 15) {
    ChannelShuffleOperatorTester()
        .batchSize(1)
        .groups(4)
        .groupChannels(groupChannels)
        .iterations(3)
        .testX8();
  }
}

TEST(CHANNEL_SHUFFLE_OP, many_groups_unit_batch) {
  for (size_t groups = 5; groups < 12; groups += 3) {
    for (size_t groupChannels = 1; groupChannels < 100; groupChannels += 15) {
      ChannelShuffleOperatorTester()
          .batchSize(1)
          .groups(groups)
          .groupChannels(groupChannels)
          .iterations(3)
          .testX8();
    }
  }
}

TEST(CHANNEL_SHUFFLE_OP, two_groups_small_batch) {
  for (size_t groupChannels = 1; groupChannels < 100; groupChannels += 15) {
    ChannelShuffleOperatorTester()
        .batchSize(3)
        .groups(2)
        .groupChannels(groupChannels)
        .iterations(3)
        .testX8();
  }
}

TEST(CHANNEL_SHUFFLE_OP, three_groups_small_batch) {
  for (size_t groupChannels = 1; groupChannels < 100; groupChannels += 15) {
    ChannelShuffleOperatorTester()
        .batchSize(3)
        .groups(3)
        .groupChannels(groupChannels)
        .iterations(3)
        .testX8();
  }
}

TEST(CHANNEL_SHUFFLE_OP, four_groups_small_batch) {
  for (size_t groupChannels = 1; groupChannels < 100; groupChannels += 15) {
    ChannelShuffleOperatorTester()
        .batchSize(3)
        .groups(4)
        .groupChannels(groupChannels)
        .iterations(3)
        .testX8();
  }
}

TEST(CHANNEL_SHUFFLE_OP, many_groups_small_batch) {
  for (size_t groups = 5; groups < 12; groups += 3) {
    for (size_t groupChannels = 1; groupChannels < 100; groupChannels += 15) {
      ChannelShuffleOperatorTester()
          .batchSize(3)
          .groups(groups)
          .groupChannels(groupChannels)
          .iterations(3)
          .testX8();
    }
  }
}

TEST(CHANNEL_SHUFFLE_OP, two_groups_small_batch_with_input_stride) {
  for (size_t groupChannels = 1; groupChannels < 100; groupChannels += 15) {
    ChannelShuffleOperatorTester()
        .batchSize(3)
        .groups(2)
        .groupChannels(groupChannels)
        .inputStride(511)
        .iterations(3)
        .testX8();
  }
}

TEST(CHANNEL_SHUFFLE_OP, three_groups_small_batch_with_input_stride) {
  for (size_t groupChannels = 1; groupChannels < 100; groupChannels += 15) {
    ChannelShuffleOperatorTester()
        .batchSize(3)
        .groups(3)
        .groupChannels(groupChannels)
        .inputStride(511)
        .iterations(3)
        .testX8();
  }
}

TEST(CHANNEL_SHUFFLE_OP, four_groups_small_batch_with_input_stride) {
  for (size_t groupChannels = 1; groupChannels < 100; groupChannels += 15) {
    ChannelShuffleOperatorTester()
        .batchSize(3)
        .groups(4)
        .groupChannels(groupChannels)
        .inputStride(511)
        .iterations(3)
        .testX8();
  }
}

TEST(CHANNEL_SHUFFLE_OP, many_groups_small_batch_with_input_stride) {
  for (size_t groups = 5; groups < 12; groups += 3) {
    for (size_t groupChannels = 1; groupChannels < 100; groupChannels += 15) {
      ChannelShuffleOperatorTester()
          .batchSize(3)
          .groups(groups)
          .groupChannels(groupChannels)
          .inputStride(1007)
          .iterations(3)
          .testX8();
    }
  }
}

TEST(CHANNEL_SHUFFLE_OP, two_groups_small_batch_with_output_stride) {
  for (size_t groupChannels = 1; groupChannels < 100; groupChannels += 15) {
    ChannelShuffleOperatorTester()
        .batchSize(3)
        .groups(2)
        .groupChannels(groupChannels)
        .outputStride(513)
        .iterations(3)
        .testX8();
  }
}

TEST(CHANNEL_SHUFFLE_OP, three_groups_small_batch_with_output_stride) {
  for (size_t groupChannels = 1; groupChannels < 100; groupChannels += 15) {
    ChannelShuffleOperatorTester()
        .batchSize(3)
        .groups(3)
        .groupChannels(groupChannels)
        .outputStride(513)
        .iterations(3)
        .testX8();
  }
}

TEST(CHANNEL_SHUFFLE_OP, four_groups_small_batch_with_output_stride) {
  for (size_t groupChannels = 1; groupChannels < 100; groupChannels += 15) {
    ChannelShuffleOperatorTester()
        .batchSize(3)
        .groups(4)
        .groupChannels(groupChannels)
        .outputStride(513)
        .iterations(3)
        .testX8();
  }
}

TEST(CHANNEL_SHUFFLE_OP, many_groups_small_batch_with_output_stride) {
  for (size_t groups = 5; groups < 12; groups += 3) {
    for (size_t groupChannels = 1; groupChannels < 100; groupChannels += 15) {
      ChannelShuffleOperatorTester()
          .batchSize(3)
          .groups(groups)
          .groupChannels(groupChannels)
          .outputStride(1111)
          .iterations(3)
          .testX8();
    }
  }
}

TEST(CHANNEL_SHUFFLE_OP, two_groups_small_batch_with_input_and_output_stride) {
  for (size_t groupChannels = 1; groupChannels < 100; groupChannels += 15) {
    ChannelShuffleOperatorTester()
        .batchSize(3)
        .groups(2)
        .groupChannels(groupChannels)
        .inputStride(511)
        .outputStride(513)
        .iterations(3)
        .testX8();
  }
}

TEST(
    CHANNEL_SHUFFLE_OP,
    three_groups_small_batch_with_input_and_output_stride) {
  for (size_t groupChannels = 1; groupChannels < 100; groupChannels += 15) {
    ChannelShuffleOperatorTester()
        .batchSize(3)
        .groups(3)
        .groupChannels(groupChannels)
        .inputStride(511)
        .outputStride(513)
        .iterations(3)
        .testX8();
  }
}

TEST(CHANNEL_SHUFFLE_OP, four_groups_small_batch_with_input_and_output_stride) {
  for (size_t groupChannels = 1; groupChannels < 100; groupChannels += 15) {
    ChannelShuffleOperatorTester()
        .batchSize(3)
        .groups(4)
        .groupChannels(groupChannels)
        .inputStride(511)
        .outputStride(513)
        .iterations(3)
        .testX8();
  }
}

TEST(CHANNEL_SHUFFLE_OP, many_groups_small_batch_with_input_and_output_stride) {
  for (size_t groups = 5; groups < 12; groups += 3) {
    for (size_t groupChannels = 1; groupChannels < 100; groupChannels += 15) {
      ChannelShuffleOperatorTester()
          .batchSize(3)
          .groups(groups)
          .groupChannels(groupChannels)
          .inputStride(1007)
          .outputStride(1111)
          .iterations(3)
          .testX8();
    }
  }
}
