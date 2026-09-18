/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "include/Config.h"
#include "src/CuptiPMSamplingProfiler.h"
#include "src/CuptiTimestamp.h"
#include "src/output_membuf.h"

using namespace KINETO_NAMESPACE;

namespace {

class FakeCuptiPMSamplingController final : public ICuptiPMSamplingController {
 public:
  FakeCuptiPMSamplingController(
      int32_t deviceId,
      std::vector<std::string> metricNames,
      std::vector<CuptiPMSample> samples = {})
      : deviceId_(deviceId),
        metricNames_(std::move(metricNames)),
        samples_(std::move(samples)) {}

  bool prepare() override {
    ++prepareCalls;
    return prepareResult;
  }

  void start() override {
    ++startCalls;
  }

  bool stop() override {
    ++stopCalls;
    return stopResult;
  }

  int32_t deviceId() const override {
    return deviceId_;
  }

  const std::vector<std::string>& metricNames() const override {
    return metricNames_;
  }

  std::vector<CuptiPMSample> takeSamples() override {
    ++takeSamplesCalls;
    return std::exchange(samples_, {});
  }

  bool prepareResult{true};
  bool stopResult{true};
  int prepareCalls{0};
  int startCalls{0};
  int stopCalls{0};
  int takeSamplesCalls{0};

 private:
  int32_t deviceId_;
  std::vector<std::string> metricNames_;
  std::vector<CuptiPMSample> samples_;
};

const std::set<ActivityType> kHardwareCounterActivities{
    ActivityType::HARDWARE_COUNTERS};

} // namespace

TEST(CuptiPMSamplingProfilerTest, ReportsNameAndSupportedActivity) {
  CuptiPMSamplingProfiler profiler;

  EXPECT_EQ(profiler.name(), "CUPTI PM Sampling");
  EXPECT_EQ(profiler.availableActivities(), kHardwareCounterActivities);
}

TEST(CuptiPMSamplingProfilerTest, RequiresActivityMetricsAndDevice) {
  CuptiPMSamplingProfiler profiler;

  Config validConfig;
  ASSERT_TRUE(
      validConfig.parse("CUPTI_PM_SAMPLING_METRICS = sm__cycles_active.avg\n"
                        "CUPTI_PM_SAMPLING_DEVICE_ID = 0"));
  EXPECT_EQ(
      profiler.configure(
          /*startTimeMs=*/123,
          /*durationMs=*/456,
          /*activityTypes=*/{},
          validConfig),
      nullptr);

  Config missingMetrics;
  ASSERT_TRUE(missingMetrics.parse("CUPTI_PM_SAMPLING_DEVICE_ID = 0"));
  EXPECT_EQ(
      profiler.configure(kHardwareCounterActivities, missingMetrics), nullptr);

  Config missingDevice;
  ASSERT_TRUE(
      missingDevice.parse("CUPTI_PM_SAMPLING_METRICS = sm__cycles_active.avg"));
  EXPECT_EQ(
      profiler.configure(kHardwareCounterActivities, missingDevice), nullptr);
}

TEST(CuptiPMSamplingSessionTest, ReportsPreparationFailure) {
  configureCuptiTimestampSource(false);
  auto controller = std::make_unique<FakeCuptiPMSamplingController>(
      0, std::vector<std::string>{"sm__cycles_active.avg"});
  controller->prepareResult = false;
  auto* controllerPtr = controller.get();
  CuptiPMSamplingSession session(std::move(controller));

  EXPECT_FALSE(session.prepare());
  EXPECT_EQ(controllerPtr->prepareCalls, 1);
}

TEST(CuptiPMSamplingSessionTest, InactiveStopDoesNotBuildTrace) {
  auto controller = std::make_unique<FakeCuptiPMSamplingController>(
      0,
      std::vector<std::string>{"sm__cycles_active.avg"},
      std::vector<CuptiPMSample>{CuptiPMSample{100, 120, {1.25}}});
  controller->stopResult = false;
  auto* controllerPtr = controller.get();
  CuptiPMSamplingSession session(std::move(controller));

  session.stop();

  EXPECT_EQ(controllerPtr->stopCalls, 1);
  EXPECT_EQ(controllerPtr->takeSamplesCalls, 0);
  EXPECT_EQ(session.getTraceBuffer(), nullptr);

  Config config;
  MemoryTraceLogger logger(config);
  session.processTrace(logger);
  EXPECT_TRUE(logger.traceActivities()->empty());
}

TEST(CuptiPMSamplingSessionTest, BuildsHardwareCounterActivities) {
  configureCuptiTimestampSource(false);
  auto controller = std::make_unique<FakeCuptiPMSamplingController>(
      2,
      std::vector<std::string>{"sm__cycles_active.avg", "dram__bytes_read.sum"},
      std::vector<CuptiPMSample>{
          CuptiPMSample{100, 120, {1.25, 2048.0}},
          CuptiPMSample{130, 170, {2.5, 4096.0}},
      });
  auto* controllerPtr = controller.get();
  CuptiPMSamplingSession session(std::move(controller));
  ASSERT_TRUE(session.prepare());
  session.start();
  session.stop();

  EXPECT_EQ(controllerPtr->prepareCalls, 1);
  EXPECT_EQ(controllerPtr->startCalls, 1);
  EXPECT_EQ(controllerPtr->stopCalls, 1);
  EXPECT_EQ(controllerPtr->takeSamplesCalls, 1);

  Config config;
  MemoryTraceLogger logger(config);
  session.processTrace(logger);
  const auto* loggedActivities = logger.traceActivities();
  ASSERT_EQ(loggedActivities->size(), 2);

  const auto& first = *loggedActivities->at(0);
  EXPECT_EQ(first.type(), ActivityType::HARDWARE_COUNTERS);
  EXPECT_EQ(first.name(), "CUPTI PM Sampling");
  EXPECT_EQ(first.deviceId(), 2);
  EXPECT_EQ(first.timestamp(), 100);
  EXPECT_EQ(first.duration(), 20);
  ASSERT_EQ(first.counterValues().size(), 2);
  EXPECT_EQ(first.counterValues()[0].first, "sm__cycles_active.avg");
  EXPECT_DOUBLE_EQ(first.counterValues()[0].second, 1.25);
  EXPECT_EQ(first.counterValues()[1].first, "dram__bytes_read.sum");
  EXPECT_DOUBLE_EQ(first.counterValues()[1].second, 2048.0);

  const auto& second = *loggedActivities->at(1);
  EXPECT_EQ(second.timestamp(), 130);
  EXPECT_EQ(second.duration(), 40);
  ASSERT_EQ(second.counterValues().size(), 2);
  EXPECT_DOUBLE_EQ(second.counterValues()[0].second, 2.5);
  EXPECT_DOUBLE_EQ(second.counterValues()[1].second, 4096.0);

  auto buffer = session.getTraceBuffer();
  ASSERT_NE(buffer, nullptr);
  EXPECT_EQ(buffer->span.name, "CUPTI PM Sampling");
  EXPECT_EQ(buffer->span.startTime, 100);
  EXPECT_EQ(buffer->span.endTime, 170);
  EXPECT_EQ(buffer->activities.size(), 2);
  EXPECT_EQ(session.getTraceBuffer().get(), nullptr);
}

TEST(CuptiPMSamplingSessionTest, TrimsSamplesToTraceWindow) {
  configureCuptiTimestampSource(false);
  auto controller = std::make_unique<FakeCuptiPMSamplingController>(
      0,
      std::vector<std::string>{"sm__cycles_active.avg"},
      std::vector<CuptiPMSample>{
          CuptiPMSample{90, 110, {1.0}},
          CuptiPMSample{120, 130, {2.0}},
          CuptiPMSample{140, 160, {3.0}},
      });
  CuptiPMSamplingSession session(std::move(controller));
  session.stop();

  Config config;
  MemoryTraceLogger logger(config);
  session.processTrace(logger, {}, 100, 150);

  const auto* loggedActivities = logger.traceActivities();
  ASSERT_EQ(loggedActivities->size(), 1);
  EXPECT_EQ(loggedActivities->at(0)->timestamp(), 120);
  EXPECT_EQ(loggedActivities->at(0)->duration(), 10);

  auto buffer = session.getTraceBuffer();
  ASSERT_NE(buffer, nullptr);
  EXPECT_EQ(buffer->span.startTime, 120);
  EXPECT_EQ(buffer->span.endTime, 130);
  EXPECT_EQ(buffer->activities.size(), 1);
}
