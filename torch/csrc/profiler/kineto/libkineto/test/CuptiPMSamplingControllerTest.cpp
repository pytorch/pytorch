/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <chrono>
#include <condition_variable>
#include <deque>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "src/CuptiPMSamplingController.h"
#include "src/ThrowUtil.h"

using namespace KINETO_NAMESPACE;
using namespace std::chrono_literals;

namespace {

struct DecodeBatch {
  std::vector<CuptiPMSample> samples;
  bool drained{true};
};

struct ApiCallCounts {
  int configureCalls{0};
  int startCalls{0};
  int stopCalls{0};
  int drainCalls{0};
  int disableCalls{0};
};

class MockCuptiPMSamplingApi final : public ICuptiPMSamplingApi {
 public:
  explicit MockCuptiPMSamplingApi(ApiCallCounts& calls) : calls_(calls) {}

  void configure(const CuptiPMSamplingConfig& config) override {
    ++calls_.configureCalls;
    receivedConfig = config;
    if (failConfiguration) {
      KINETO_THROW(std::runtime_error, "PM sampling configuration failed");
    }
  }

  void start() override {
    std::lock_guard<std::mutex> lock(mutex_);
    ++calls_.startCalls;
    stopped_ = false;
    if (failStart) {
      KINETO_THROW(std::runtime_error, "PM sampling start failed");
    }
  }

  bool decode(std::vector<CuptiPMSample>& samples) override {
    std::lock_guard<std::mutex> lock(mutex_);
    ++decodeCalls_;
    if (stopped_) {
      ++calls_.drainCalls;
    }
    decoded_.notify_all();
    std::deque<DecodeBatch>& batches = stopped_ ? drainBatches : runningBatches;
    // Allow samples to accumulate before a scripted worker failure.
    if ((failDecode && batches.empty()) || (stopped_ && failDrain)) {
      KINETO_THROW(std::runtime_error, "PM sampling decode failed");
    }
    if (batches.empty()) {
      return true;
    }
    const DecodeBatch& batch = batches.front();
    samples.insert(samples.end(), batch.samples.begin(), batch.samples.end());
    const bool drained = batch.drained;
    batches.pop_front();
    return drained;
  }

  void stop() override {
    std::lock_guard<std::mutex> lock(mutex_);
    ++calls_.stopCalls;
    stopped_ = true;
    if (failStop) {
      KINETO_THROW(std::runtime_error, "PM sampling stop failed");
    }
  }

  void disable() noexcept override {
    ++calls_.disableCalls;
  }

  bool waitForDecodeCalls(int count) {
    std::unique_lock<std::mutex> lock(mutex_);
    return decoded_.wait_for(lock, 5s, [&] { return decodeCalls_ >= count; });
  }

  int decodeCalls() {
    std::lock_guard<std::mutex> lock(mutex_);
    return decodeCalls_;
  }

  // Set scripts before start(); inspect results after stop() joins the worker.
  std::deque<DecodeBatch> runningBatches;
  std::deque<DecodeBatch> drainBatches;
  std::optional<CuptiPMSamplingConfig> receivedConfig;
  bool failConfiguration{false};
  bool failStart{false};
  bool failDecode{false};
  bool failStop{false};
  bool failDrain{false};

 private:
  ApiCallCounts& calls_;
  std::mutex mutex_;
  std::condition_variable decoded_;
  int decodeCalls_{0};
  bool stopped_{false};
};

const CuptiPMSamplingConfig kConfig{0, {"sm__cycles_active.avg"}, 1ms, 1s};

} // namespace

TEST(CuptiPMSamplingControllerTest, RejectsNullApi) {
  EXPECT_THROW(
      CuptiPMSamplingController(kConfig, nullptr), std::invalid_argument);
}

TEST(CuptiPMSamplingControllerTest, ForwardsConfigurationToApi) {
  const std::vector<CuptiPMSamplingConfig> configs{
      {2, {"sm__cycles_active.avg", "dram__bytes_read.sum"}, std::nullopt, 2s},
      {0, {"sm__cycles_active.avg"}, 1ms, 1s},
  };
  for (const CuptiPMSamplingConfig& config : configs) {
    SCOPED_TRACE(config.deviceId);
    ApiCallCounts calls;
    auto api = std::make_unique<MockCuptiPMSamplingApi>(calls);
    const MockCuptiPMSamplingApi& mockApi = *api;
    CuptiPMSamplingController controller(config, std::move(api));

    ASSERT_TRUE(controller.prepare());

    ASSERT_TRUE(mockApi.receivedConfig.has_value());
    EXPECT_EQ(mockApi.receivedConfig->deviceId, config.deviceId);
    EXPECT_EQ(mockApi.receivedConfig->metricNames, config.metricNames);
    EXPECT_EQ(
        mockApi.receivedConfig->samplingInterval, config.samplingInterval);
    EXPECT_EQ(mockApi.receivedConfig->lookbackWindow, config.lookbackWindow);
    EXPECT_FALSE(controller.stop());
    EXPECT_EQ(calls.disableCalls, 1);
  }
}

TEST(CuptiPMSamplingControllerTest, CleansUpWhenConfigurationFails) {
  ApiCallCounts calls;
  auto api = std::make_unique<MockCuptiPMSamplingApi>(calls);
  api->failConfiguration = true;
  const MockCuptiPMSamplingApi& mockApi = *api;
  CuptiPMSamplingController controller(kConfig, std::move(api));

  EXPECT_FALSE(controller.prepare());

  ASSERT_TRUE(mockApi.receivedConfig.has_value());
  EXPECT_EQ(calls.disableCalls, 1);
  EXPECT_FALSE(controller.stop());
  EXPECT_EQ(calls.disableCalls, 1);
}

TEST(CuptiPMSamplingControllerTest, RejectsInvalidConfigBeforeCallingApi) {
  const std::vector<CuptiPMSamplingConfig> configs{
      {-1, {"sm__cycles_active.avg"}, 1ms, 1s},
      {0, {}, 1ms, 1s},
      {0, {""}, 1ms, 1s},
      {0, {"sm__cycles_active.avg", ""}, 1ms, 1s},
  };
  for (const CuptiPMSamplingConfig& config : configs) {
    SCOPED_TRACE(testing::PrintToString(config.metricNames));
    ApiCallCounts calls;
    auto api = std::make_unique<MockCuptiPMSamplingApi>(calls);
    const MockCuptiPMSamplingApi& mockApi = *api;
    CuptiPMSamplingController controller(config, std::move(api));

    EXPECT_FALSE(controller.prepare());
    controller.start();

    EXPECT_EQ(calls.configureCalls, 0);
    EXPECT_FALSE(mockApi.receivedConfig.has_value());
    EXPECT_EQ(calls.startCalls, 0);
    EXPECT_FALSE(controller.stop());
    EXPECT_EQ(calls.disableCalls, 0);
  }
}

TEST(CuptiPMSamplingControllerTest, RejectsPreparationUntilStopped) {
  ApiCallCounts calls;
  auto api = std::make_unique<MockCuptiPMSamplingApi>(calls);
  CuptiPMSamplingController controller(kConfig, std::move(api));

  ASSERT_TRUE(controller.prepare());
  EXPECT_FALSE(controller.prepare());
  EXPECT_EQ(calls.configureCalls, 1);
  EXPECT_EQ(calls.disableCalls, 0);

  EXPECT_FALSE(controller.stop());
  EXPECT_EQ(calls.disableCalls, 1);
  EXPECT_TRUE(controller.prepare());
  EXPECT_EQ(calls.configureCalls, 2);
  EXPECT_FALSE(controller.stop());
  EXPECT_EQ(calls.disableCalls, 2);
}

TEST(CuptiPMSamplingControllerTest, IgnoresStartWhileActive) {
  ApiCallCounts calls;
  auto api = std::make_unique<MockCuptiPMSamplingApi>(calls);
  MockCuptiPMSamplingApi& mockApi = *api;
  CuptiPMSamplingController controller(kConfig, std::move(api));

  ASSERT_TRUE(controller.prepare());
  controller.start();
  EXPECT_TRUE(mockApi.waitForDecodeCalls(1));
  EXPECT_NO_THROW(controller.start());
  EXPECT_TRUE(controller.stop());

  EXPECT_EQ(calls.startCalls, 1);
  EXPECT_EQ(calls.stopCalls, 1);
  EXPECT_EQ(calls.drainCalls, 1);
  EXPECT_EQ(calls.disableCalls, 1);
}

TEST(CuptiPMSamplingControllerTest, RequiresPreparationToRestartAfterStop) {
  ApiCallCounts calls;
  auto api = std::make_unique<MockCuptiPMSamplingApi>(calls);
  MockCuptiPMSamplingApi& mockApi = *api;
  CuptiPMSamplingController controller(kConfig, std::move(api));

  ASSERT_TRUE(controller.prepare());
  controller.start();
  EXPECT_TRUE(mockApi.waitForDecodeCalls(1));
  EXPECT_TRUE(controller.stop());
  const int decodeCalls = mockApi.decodeCalls();

  EXPECT_NO_THROW(controller.start());
  EXPECT_FALSE(controller.stop());

  EXPECT_EQ(calls.configureCalls, 1);
  EXPECT_EQ(calls.startCalls, 1);
  EXPECT_EQ(calls.stopCalls, 1);
  EXPECT_EQ(calls.drainCalls, 1);
  EXPECT_EQ(calls.disableCalls, 1);
  EXPECT_EQ(mockApi.decodeCalls(), decodeCalls);
}

TEST(CuptiPMSamplingControllerTest, FiltersWorkerSamplesAndTransfersOwnership) {
  ApiCallCounts calls;
  auto api = std::make_unique<MockCuptiPMSamplingApi>(calls);
  api->runningBatches = {
      {{{100, 110, {1.0}}}, false},
      {{{120, 130, {2.0}},
        {140, 150, {}},
        {160, 170, {3.0, 4.0}},
        {190, 180, {5.0}},
        {200, 210, {std::numeric_limits<double>::quiet_NaN()}},
        {220, 230, {std::numeric_limits<double>::infinity()}},
        {240, 250, {-std::numeric_limits<double>::infinity()}},
        {260, 260, {0.0}}},
       true},
  };
  MockCuptiPMSamplingApi& mockApi = *api;
  CuptiPMSamplingController controller(kConfig, std::move(api));

  ASSERT_TRUE(controller.prepare());
  controller.start();
  EXPECT_TRUE(mockApi.waitForDecodeCalls(2));
  EXPECT_TRUE(controller.stop());

  const std::vector<CuptiPMSample> samples = controller.takeSamples();
  ASSERT_EQ(samples.size(), 2);
  EXPECT_EQ(samples[0].rawStartTimestamp, 120);
  EXPECT_EQ(samples[0].rawEndTimestamp, 130);
  EXPECT_EQ(samples[0].values, (std::vector<double>{2.0}));
  EXPECT_EQ(samples[1].rawStartTimestamp, 260);
  EXPECT_EQ(samples[1].rawEndTimestamp, 260);
  EXPECT_EQ(samples[1].values, (std::vector<double>{0.0}));
  EXPECT_TRUE(controller.takeSamples().empty());
}

TEST(CuptiPMSamplingControllerTest, DrainsUntilAllBatchesAreDecoded) {
  ApiCallCounts calls;
  auto api = std::make_unique<MockCuptiPMSamplingApi>(calls);
  api->drainBatches = {
      {{{100, 110, {1.0}}}, false},
      {{{120, 130, {2.0}}}, false},
      {{{140, 150, {3.0}}}, true},
  };
  MockCuptiPMSamplingApi& mockApi = *api;
  CuptiPMSamplingController controller(kConfig, std::move(api));

  ASSERT_TRUE(controller.prepare());
  controller.start();
  EXPECT_TRUE(mockApi.waitForDecodeCalls(1));
  EXPECT_TRUE(controller.stop());

  EXPECT_EQ(calls.startCalls, 1);
  EXPECT_EQ(calls.stopCalls, 1);
  EXPECT_EQ(calls.drainCalls, 3);
  EXPECT_EQ(calls.disableCalls, 1);
  const std::vector<CuptiPMSample> samples = controller.takeSamples();
  ASSERT_EQ(samples.size(), 2);
  EXPECT_EQ(samples[0].rawStartTimestamp, 120);
  EXPECT_EQ(samples[0].rawEndTimestamp, 130);
  EXPECT_EQ(samples[0].values, (std::vector<double>{2.0}));
  EXPECT_EQ(samples[1].rawStartTimestamp, 140);
  EXPECT_EQ(samples[1].rawEndTimestamp, 150);
  EXPECT_EQ(samples[1].values, (std::vector<double>{3.0}));

  EXPECT_FALSE(controller.stop());
  EXPECT_EQ(calls.stopCalls, 1);
  EXPECT_EQ(calls.drainCalls, 3);
  EXPECT_EQ(calls.disableCalls, 1);
}

TEST(CuptiPMSamplingControllerTest, SkipsFinalDrainAfterWorkerFailure) {
  ApiCallCounts calls;
  auto api = std::make_unique<MockCuptiPMSamplingApi>(calls);
  api->failDecode = true;
  MockCuptiPMSamplingApi& mockApi = *api;
  CuptiPMSamplingController controller(kConfig, std::move(api));

  ASSERT_TRUE(controller.prepare());
  controller.start();
  EXPECT_TRUE(mockApi.waitForDecodeCalls(1));
  EXPECT_TRUE(controller.stop());

  EXPECT_EQ(mockApi.decodeCalls(), 1);
  EXPECT_EQ(calls.stopCalls, 1);
  EXPECT_EQ(calls.drainCalls, 0);
  EXPECT_EQ(calls.disableCalls, 1);
  EXPECT_TRUE(controller.takeSamples().empty());
}

TEST(CuptiPMSamplingControllerTest, ResetsSessionStateAfterWorkerFailure) {
  ApiCallCounts calls;
  auto api = std::make_unique<MockCuptiPMSamplingApi>(calls);
  api->runningBatches = {{{{100, 110, {1.0}}, {120, 130, {2.0}}}, false}};
  api->failDecode = true;
  MockCuptiPMSamplingApi& mockApi = *api;
  CuptiPMSamplingController controller(kConfig, std::move(api));

  ASSERT_TRUE(controller.prepare());
  controller.start();
  EXPECT_TRUE(mockApi.waitForDecodeCalls(2));
  EXPECT_TRUE(controller.stop());
  EXPECT_EQ(mockApi.decodeCalls(), 2);
  EXPECT_TRUE(mockApi.runningBatches.empty());
  EXPECT_EQ(calls.drainCalls, 0);
  EXPECT_EQ(calls.disableCalls, 1);

  // Leave the first session's valid sample buffered for prepare() to clear.
  mockApi.failDecode = false;
  mockApi.drainBatches = {
      {{{300, 310, {3.0}}, {320, 330, {4.0}}}, false},
      {{{340, 350, {5.0}}}, true},
  };
  ASSERT_TRUE(controller.prepare());
  controller.start();
  EXPECT_TRUE(mockApi.waitForDecodeCalls(3));
  EXPECT_TRUE(controller.stop());

  EXPECT_EQ(calls.configureCalls, 2);
  EXPECT_EQ(calls.startCalls, 2);
  EXPECT_EQ(calls.stopCalls, 2);
  EXPECT_EQ(calls.drainCalls, 2);
  EXPECT_EQ(calls.disableCalls, 2);
  const std::vector<CuptiPMSample> samples = controller.takeSamples();
  ASSERT_EQ(samples.size(), 2);
  EXPECT_EQ(samples[0].rawStartTimestamp, 320);
  EXPECT_EQ(samples[0].rawEndTimestamp, 330);
  EXPECT_EQ(samples[0].values, (std::vector<double>{4.0}));
  EXPECT_EQ(samples[1].rawStartTimestamp, 340);
  EXPECT_EQ(samples[1].rawEndTimestamp, 350);
  EXPECT_EQ(samples[1].values, (std::vector<double>{5.0}));
}

TEST(CuptiPMSamplingControllerTest, DestructionStopsActiveSession) {
  ApiCallCounts calls;
  {
    auto api = std::make_unique<MockCuptiPMSamplingApi>(calls);
    api->drainBatches = {
        {{{100, 110, {1.0}}}, false},
        {{{120, 130, {2.0}}}, true},
    };
    MockCuptiPMSamplingApi& mockApi = *api;
    CuptiPMSamplingController controller(kConfig, std::move(api));

    ASSERT_TRUE(controller.prepare());
    controller.start();
    EXPECT_TRUE(mockApi.waitForDecodeCalls(1));
    EXPECT_EQ(calls.stopCalls, 0);
    EXPECT_EQ(calls.disableCalls, 0);
  }

  EXPECT_EQ(calls.stopCalls, 1);
  EXPECT_EQ(calls.drainCalls, 2);
  EXPECT_EQ(calls.disableCalls, 1);
}

TEST(CuptiPMSamplingControllerTest, DisablesAfterStopOrDrainFailure) {
  for (const bool failStop : {true, false}) {
    SCOPED_TRACE(failStop);
    ApiCallCounts calls;
    {
      auto api = std::make_unique<MockCuptiPMSamplingApi>(calls);
      api->failStop = failStop;
      api->failDrain = !failStop;
      MockCuptiPMSamplingApi& mockApi = *api;
      CuptiPMSamplingController controller(kConfig, std::move(api));

      ASSERT_TRUE(controller.prepare());
      controller.start();
      EXPECT_TRUE(mockApi.waitForDecodeCalls(1));
      EXPECT_TRUE(controller.stop());
      EXPECT_EQ(calls.stopCalls, 1);
      EXPECT_EQ(calls.drainCalls, failStop ? 0 : 1);
      EXPECT_EQ(calls.disableCalls, 1);
      EXPECT_FALSE(controller.stop());
    }

    EXPECT_EQ(calls.stopCalls, 1);
    EXPECT_EQ(calls.drainCalls, failStop ? 0 : 1);
    EXPECT_EQ(calls.disableCalls, 1);
  }
}

TEST(
    CuptiPMSamplingControllerTest,
    DisablesWithoutStartingWorkerOnStartFailure) {
  ApiCallCounts calls;
  auto api = std::make_unique<MockCuptiPMSamplingApi>(calls);
  api->failStart = true;
  MockCuptiPMSamplingApi& mockApi = *api;
  CuptiPMSamplingController controller(kConfig, std::move(api));

  ASSERT_TRUE(controller.prepare());
  EXPECT_NO_THROW(controller.start());
  EXPECT_EQ(calls.startCalls, 1);
  EXPECT_EQ(calls.disableCalls, 1);
  EXPECT_FALSE(controller.stop());
  EXPECT_EQ(calls.stopCalls, 0);
  EXPECT_EQ(mockApi.decodeCalls(), 0);
  EXPECT_EQ(calls.disableCalls, 1);
}
