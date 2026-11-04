/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "include/Config.h"
#include "src/CuptiPMSamplingProfiler.h"
#include "src/CuptiTimestamp.h"
#include "test/MockCuptiPMSampling.h"

using namespace KINETO_NAMESPACE;
using namespace libkineto::test;

namespace {

class CuptiPMSamplingIntegrationTest : public testing::Test {
 protected:
  void SetUp() override {
    mockCupti() = MockCuptiState{};
  }
};

} // namespace

TEST_F(CuptiPMSamplingIntegrationTest, ProfilerForwardsConfiguration) {
  Config config;
  ASSERT_TRUE(config.parse(
      "PERFORMANCE_METRICS_DEVICE_ID=2\n"
      "PERFORMANCE_METRICS=sm__cycles_active.avg,dram__bytes_read.sum\n"
      "PERFORMANCE_METRICS_SAMPLING_INTERVAL_MS=0.25\n"
      "PERFORMANCE_METRICS_LOOKBACK_WINDOW_MS=2000"));
  configureCuptiTimestampSource(false);
  CuptiPMSamplingProfiler profiler;

  auto session = profiler.configure({ActivityType::HARDWARE_COUNTERS}, config);

  ASSERT_NE(session, nullptr);
  EXPECT_EQ(mockCupti().configMetricNames, config.performanceMetricNames());
  EXPECT_EQ(mockCupti().enabledDeviceIndex, 2);
  EXPECT_EQ(mockCupti().samplingInterval, 250'000);
  EXPECT_EQ(mockCupti().maxSamples, 8'000);
}
