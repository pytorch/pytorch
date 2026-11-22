/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "src/DeviceProperties.h"

using namespace KINETO_NAMESPACE;

class OccupancyMetricsTest : public ::testing::Test {};

#ifdef HAS_CUPTI

// Verify all cudaOccResult fields are mapped to OccupancyMetrics
TEST_F(OccupancyMetricsTest, AllFieldsPopulated) {
  if (smCount(0) == 0) {
    GTEST_SKIP() << "No GPU available";
  }

  CUpti_ActivityKernelType kernel = {};
  kernel.deviceId = 0;
  kernel.registersPerThread = 32;
  kernel.staticSharedMemory = 0;
  kernel.dynamicSharedMemory = 0;
  kernel.blockX = 256;
  kernel.blockY = 1;
  kernel.blockZ = 1;
  kernel.gridX = 100;
  kernel.gridY = 1;
  kernel.gridZ = 1;

  OccupancyMetrics metrics = computeOccupancyMetrics(kernel);

  // All fields from cudaOccResult should be populated (non-default)
  EXPECT_NE(metrics.occupancy, -1.0f);
  EXPECT_NE(metrics.result.activeBlocksPerMultiprocessor, 0);
  // limitingFactors can legitimately be 0 if nothing is limiting
  EXPECT_NE(metrics.result.blockLimitRegs, 0);
  EXPECT_NE(metrics.result.blockLimitSharedMem, 0);
  EXPECT_NE(metrics.result.blockLimitWarps, 0);
  EXPECT_NE(metrics.result.blockLimitBlocks, 0);
  // blockLimitBarriers can be 0 if no barriers used
  EXPECT_NE(metrics.result.allocatedRegistersPerBlock, 0);
  // allocatedSharedMemPerBlock can be 0 if no shared mem used
  EXPECT_EQ(metrics.result.partitionedGCConfig, PARTITIONED_GC_OFF);
}

#endif // HAS_CUPTI
