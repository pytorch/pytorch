/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>
#include <cupti_pmsampling.h>

#include "src/CuptiPMSamplingApi.h"
#include "src/Logger.h"
#include "src/LoggerCollector.h"
#include "test/MockCuptiPMSampling.h"

using namespace KINETO_NAMESPACE;
using namespace libkineto::test;
using namespace std::chrono_literals;

namespace {

void clearCalls() {
  mockCupti().calls.clear();
}

void expectCalls(std::initializer_list<std::string> expected) {
  EXPECT_EQ(mockCupti().calls, std::vector<std::string>(expected));
}

CuptiPMSamplingConfig makeConfig(
    std::optional<std::chrono::nanoseconds> samplingInterval = 500us,
    int32_t deviceId = 0,
    std::vector<std::string> metricNames = {"sm__cycles_active.avg"},
    std::chrono::nanoseconds lookbackWindow = 1s) {
  return CuptiPMSamplingConfig{
      deviceId, std::move(metricNames), samplingInterval, lookbackWindow};
}

void configureForDevice(
    int major,
    int minor,
    std::optional<std::chrono::nanoseconds> samplingInterval) {
  mockCupti().major = major;
  mockCupti().minor = minor;
  CuptiPMSamplingApi api;
  api.configure(makeConfig(samplingInterval));
  api.disable();
}

class CuptiPMSamplingApiTest : public testing::Test {
 protected:
  void SetUp() override {
    mockCupti() = MockCuptiState{};
  }
};

} // namespace

TEST_F(CuptiPMSamplingApiTest, RequiresConfigurationForSamplingOperations) {
  CuptiPMSamplingApi api;
  std::vector<CuptiPMSample> samples;

  EXPECT_THROW(api.start(), std::runtime_error);
  EXPECT_THROW(api.decode(samples), std::runtime_error);
  EXPECT_THROW(api.stop(), std::runtime_error);
  EXPECT_NO_THROW(api.disable());
  EXPECT_TRUE(samples.empty());
  EXPECT_TRUE(mockCupti().calls.empty());
}

TEST_F(CuptiPMSamplingApiTest, ConfiguresDeviceMetricsAndBuffers) {
  const auto config = makeConfig(
      250us,
      /*deviceId=*/2,
      {"sm__cycles_active.avg", "dram__bytes_read.sum"},
      2s);
  CuptiPMSamplingApi api;

  api.configure(config);

  EXPECT_EQ(mockCupti().configMetricNames, config.metricNames);
  EXPECT_EQ(mockCupti().enabledDeviceIndex, 2);
  EXPECT_EQ(mockCupti().hardwareBufferSize, 64 * 1024 * 1024);
  EXPECT_EQ(mockCupti().samplingInterval, 250'000);
  EXPECT_EQ(
      mockCupti().triggerMode,
      CUPTI_PM_SAMPLING_TRIGGER_MODE_GPU_TIME_INTERVAL);
  EXPECT_EQ(
      mockCupti().appendMode,
      CUPTI_PM_SAMPLING_HARDWARE_BUFFER_APPEND_MODE_KEEP_LATEST);
  EXPECT_EQ(mockCupti().counterDataMetricNames, config.metricNames);
  EXPECT_EQ(mockCupti().maxSamples, 8'000);
  api.disable();
}

TEST_F(CuptiPMSamplingApiTest, KeepsAtLeastOneSample) {
  CuptiPMSamplingApi api;

  api.configure(makeConfig(
      2ms,
      /*deviceId=*/0,
      {"sm__cycles_active.avg"},
      1ms));

  EXPECT_EQ(mockCupti().maxSamples, 1);
  api.disable();
}

TEST_F(CuptiPMSamplingApiTest, SizesGa100CapacityFromPeakClock) {
  mockCupti().major = 8;
  mockCupti().minor = 0;
  CuptiPMSamplingApi api;
  api.configure(makeConfig(
      std::nullopt,
      /*deviceId=*/2,
      {"sm__cycles_active.avg"},
      2s));

  EXPECT_EQ(mockCupti().clockRateDevice, 2);
  EXPECT_EQ(
      mockCupti().triggerMode,
      CUPTI_PM_SAMPLING_TRIGGER_MODE_GPU_SYSCLK_INTERVAL);
  EXPECT_EQ(mockCupti().samplingInterval, 1'000'000);
  // At the mock 1.5 GHz maximum clock, 1M cycles yields 3K samples in 2s.
  EXPECT_EQ(mockCupti().maxSamples, 3'000);
  api.disable();
}

TEST_F(CuptiPMSamplingApiTest, IgnoresExplicitTimeIntervalsOnGa100) {
  for (const std::chrono::nanoseconds interval :
       {0ns, 1ns, 500'000ns, 1'000'000ns}) {
    SCOPED_TRACE(interval.count());
    EXPECT_NO_THROW(configureForDevice(8, 0, interval));
    EXPECT_EQ(
        mockCupti().triggerMode,
        CUPTI_PM_SAMPLING_TRIGGER_MODE_GPU_SYSCLK_INTERVAL);
    EXPECT_EQ(mockCupti().samplingInterval, 1'000'000);
    EXPECT_EQ(mockCupti().maxSamples, 1'500);
  }
}

#if !USE_GOOGLE_LOG
TEST_F(CuptiPMSamplingApiTest, WarnsAndContinuesWithExplicitGa100Interval) {
  mockCupti().major = 8;
  mockCupti().minor = 0;
  CuptiPMSamplingApi api;
  LoggerCollector collector;
  const int previousSeverity = Logger::severityLevel();
  Logger::setSeverityLevel(LoggerOutputType::WARNING);
  Logger::addLoggerObserver(&collector);

  EXPECT_NO_THROW(api.configure(makeConfig(1ms)));

  Logger::removeLoggerObserver(&collector);
  Logger::setSeverityLevel(previousSeverity);
  auto messages = collector.extractCollectorMetadata();
  EXPECT_TRUE(messages[LoggerOutputType::ERROR].empty());
  const auto& warnings = messages[LoggerOutputType::WARNING];
  ASSERT_EQ(warnings.size(), 1);
  EXPECT_NE(
      warnings.front().find("PERFORMANCE_METRICS_SAMPLING_INTERVAL_MS"),
      std::string::npos);
  EXPECT_NE(warnings.front().find("ignored"), std::string::npos);
  EXPECT_NE(warnings.front().find("1,000,000-cycle"), std::string::npos);
  EXPECT_EQ(mockCupti().clockRateDevice, 0);
  EXPECT_EQ(mockCupti().samplingInterval, 1'000'000);
  EXPECT_EQ(mockCupti().maxSamples, 1'500);
  api.disable();
}
#endif

TEST_F(CuptiPMSamplingApiTest, CleansUpAfterGa100ClockQueryFailure) {
  mockCupti().major = 8;
  mockCupti().minor = 0;
  mockCupti().clockRateResult = cudaErrorInvalidValue;
  CuptiPMSamplingApi api;

  EXPECT_THROW(api.configure(makeConfig(std::nullopt)), std::runtime_error);

  clearCalls();
  api.disable();
  expectCalls({"profilerDeInitialize"});
}

TEST_F(CuptiPMSamplingApiTest, RejectsNonpositiveGa100ClockRates) {
  for (const int clockRate : {0, -1}) {
    SCOPED_TRACE(clockRate);
    mockCupti().clockRate = clockRate;
    EXPECT_THROW(configureForDevice(8, 0, std::nullopt), std::runtime_error);
  }
}

TEST_F(CuptiPMSamplingApiTest, UsesDefaultTimeIntervalOnGa10xAndNewer) {
  for (const auto& [major, minor] : {std::pair{8, 6}, std::pair{9, 0}}) {
    SCOPED_TRACE(testing::Message() << major << "." << minor);
    configureForDevice(major, minor, std::nullopt);

    EXPECT_EQ(mockCupti().samplingInterval, 1'000'000);
    EXPECT_EQ(mockCupti().maxSamples, 1'000);
  }
}

TEST_F(CuptiPMSamplingApiTest, UsesRequestedTimeIntervalOnGa10xAndNewer) {
  for (const auto& [major, minor] : {std::pair{8, 6}, std::pair{9, 0}}) {
    SCOPED_TRACE(testing::Message() << major << "." << minor);
    configureForDevice(major, minor, 500us);

    EXPECT_EQ(
        mockCupti().triggerMode,
        CUPTI_PM_SAMPLING_TRIGGER_MODE_GPU_TIME_INTERVAL);
    EXPECT_EQ(mockCupti().samplingInterval, 500'000);
    EXPECT_EQ(mockCupti().clockRateDevice, -1);
  }
}

TEST_F(CuptiPMSamplingApiTest, RejectsUnsupportedComputeCapabilities) {
  for (const auto& [major, minor] : {std::pair{7, 5}, std::pair{8, 5}}) {
    SCOPED_TRACE(testing::Message() << major << "." << minor);
    EXPECT_THROW(configureForDevice(major, minor, 500us), std::runtime_error);
  }
}

TEST_F(CuptiPMSamplingApiTest, RejectsNonpositiveTimeIntervals) {
  EXPECT_THROW(configureForDevice(8, 6, 0ns), std::runtime_error);
}

TEST_F(CuptiPMSamplingApiTest, RejectsNonpositiveLookbackWindows) {
  CuptiPMSamplingApi api;
  EXPECT_THROW(
      api.configure(makeConfig(
          500us,
          /*deviceId=*/0,
          {"sm__cycles_active.avg"},
          0ns)),
      std::runtime_error);
}

TEST_F(CuptiPMSamplingApiTest, RejectsSampleCapacityAboveCuptiLimit) {
  CuptiPMSamplingApi api;
  constexpr auto kTooManySamples =
      static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) + 1;
  EXPECT_THROW(
      api.configure(makeConfig(
          1ns,
          /*deviceId=*/0,
          {"sm__cycles_active.avg"},
          std::chrono::nanoseconds{kTooManySamples})),
      std::runtime_error);
}

TEST_F(CuptiPMSamplingApiTest, RejectsCounterDataImageAboveMemoryLimit) {
  mockCupti().counterDataSize = 256 * 1024 * 1024 + 1;
  CuptiPMSamplingApi api;

  EXPECT_THROW(api.configure(makeConfig()), std::runtime_error);

  clearCalls();
  api.disable();
  expectCalls({"samplingDisable", "hostDeinitialize", "profilerDeInitialize"});
}

TEST_F(CuptiPMSamplingApiTest, RejectsMultipassConfigurationBeforeEnabling) {
  mockCupti().numPasses = 2;
  CuptiPMSamplingApi api;

  EXPECT_THROW(api.configure(makeConfig()), std::runtime_error);

  clearCalls();
  api.disable();
  expectCalls({"hostDeinitialize", "profilerDeInitialize"});
}

TEST_F(CuptiPMSamplingApiTest, CleansUpAfterCuptiConfigurationFailure) {
  mockCupti().setConfigResult = CUPTI_ERROR_UNKNOWN;
  CuptiPMSamplingApi api;

  EXPECT_THROW(api.configure(makeConfig()), std::runtime_error);

  clearCalls();
  api.disable();
  expectCalls({"samplingDisable", "hostDeinitialize", "profilerDeInitialize"});
}

TEST_F(CuptiPMSamplingApiTest, RejectsReconfigurationUntilDisabled) {
  CuptiPMSamplingApi api;
  api.configure(makeConfig());

  EXPECT_THROW(
      api.configure(makeConfig(750us, 1, {"dram__bytes_read.sum"})),
      std::runtime_error);

  api.disable();
  api.configure(makeConfig(750us, 1, {"dram__bytes_read.sum"}));
  EXPECT_EQ(
      mockCupti().configMetricNames,
      (std::vector<std::string>{"dram__bytes_read.sum"}));
  api.disable();
}

TEST_F(CuptiPMSamplingApiTest, StartsAndStopsConfiguredSampler) {
  CuptiPMSamplingApi api;
  api.configure(makeConfig());
  clearCalls();

  api.start();
  api.stop();

  expectCalls({"samplingStart", "samplingStop"});
  api.disable();
}

TEST_F(CuptiPMSamplingApiTest, DecodesCompletedSamplesAndAppendsThem) {
  const auto config = makeConfig(
      500us,
      /*deviceId=*/0,
      {"sm__cycles_active.avg", "dram__bytes_read.sum"});
  mockCupti().decodedSamples = {
      MockSample{100, 120, {1.25, 2048.0}},
      MockSample{130, 170, {2.5, 4096.0}},
  };
  CuptiPMSamplingApi api;
  api.configure(config);
  std::vector<CuptiPMSample> samples{
      CuptiPMSample{1, 2, std::vector<double>{3.0}}};
  clearCalls();

  EXPECT_TRUE(api.decode(samples));

  expectCalls({"counterDataImageInitialize"});
  ASSERT_EQ(samples.size(), 3);
  EXPECT_EQ(samples[0].rawStartTimestamp, 1);
  EXPECT_EQ(samples[0].rawEndTimestamp, 2);
  EXPECT_EQ(samples[0].values, (std::vector<double>{3.0}));
  EXPECT_EQ(samples[1].rawStartTimestamp, 100);
  EXPECT_EQ(samples[1].rawEndTimestamp, 120);
  EXPECT_EQ(samples[1].values, (std::vector<double>{1.25, 2048.0}));
  EXPECT_EQ(samples[2].rawStartTimestamp, 130);
  EXPECT_EQ(samples[2].rawEndTimestamp, 170);
  EXPECT_EQ(samples[2].values, (std::vector<double>{2.5, 4096.0}));
  api.disable();
}

TEST_F(CuptiPMSamplingApiTest, ReportsFullCounterDataAsUndrained) {
  mockCupti().decodedSamples = {
      MockSample{100, 120, {1.25}},
  };
  CuptiPMSamplingApi api;
  api.configure(makeConfig());
  std::vector<CuptiPMSample> samples;

  mockCupti().decodeStopReason =
      CUPTI_PM_SAMPLING_DECODE_STOP_REASON_COUNTER_DATA_FULL;
  EXPECT_FALSE(api.decode(samples));
  ASSERT_EQ(samples.size(), 1);
  EXPECT_EQ(samples[0].rawStartTimestamp, 100);
  EXPECT_EQ(samples[0].rawEndTimestamp, 120);
  EXPECT_EQ(samples[0].values, (std::vector<double>{1.25}));
  api.disable();
}

TEST_F(CuptiPMSamplingApiTest, DisableIsOrderedAndIdempotent) {
  CuptiPMSamplingApi api;
  api.configure(makeConfig());
  clearCalls();

  api.disable();

  expectCalls({"samplingDisable", "hostDeinitialize", "profilerDeInitialize"});
  clearCalls();
  api.disable();
  EXPECT_TRUE(mockCupti().calls.empty());
  EXPECT_THROW(api.start(), std::runtime_error);
}

TEST_F(CuptiPMSamplingApiTest, DestructorDisablesConfiguredSampler) {
  {
    CuptiPMSamplingApi api;
    api.configure(makeConfig());
    clearCalls();
  }

  expectCalls({"samplingDisable", "hostDeinitialize", "profilerDeInitialize"});
}
