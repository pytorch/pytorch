/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <fmt/format.h>
#include <gtest/gtest.h>
#include <cstdlib>

#include <chrono>
#include <fstream>
#include <string>

#include "include/Config.h"
#include "src/ActivityProfilerController.h"
#include "test/TestUtils.h"

using namespace std::chrono;
using namespace KINETO_NAMESPACE;
using namespace libkineto::test;

namespace {

bool traceFileHasContent(const std::string& filename) {
  std::ifstream file(filename, std::ios::binary | std::ios::ate);
  return file.is_open() && file.tellg() > 0;
}

} // namespace

TEST(ActivityProfilerController, PrepareTraceClearsPendingAsyncRequest) {
  auto traceFile = createTempTraceFile("libkineto_test", ".json");
  Config asyncCfg;
  bool success = asyncCfg.parse(fmt::format(
      R"CFG(
    PROFILE_START_ITERATION = 5
    ACTIVITIES_WARMUP_ITERATIONS = 1
    ACTIVITIES_ITERATIONS = 2
    ACTIVITIES_DURATION_SECS = 1
    ACTIVITIES_LOG_FILE = {}
  )CFG",
      traceFile.path()));
  ASSERT_TRUE(success);

  // Start an async request via acceptConfig -- populate asyncRequestConfig_
  ActivityProfilerController controller(ConfigLoader::instance(), true);
  controller.asyncStep();
  controller.acceptConfig(asyncCfg);
  EXPECT_FALSE(controller.isActive());

  // Start a sync trace before the async request starts
  Config syncCfg;
  syncCfg.setClientDefaults();
  syncCfg.validate(system_clock::now());
  controller.syncPrepareTrace(syncCfg);
  EXPECT_TRUE(controller.isActive());

  controller.syncStartTrace();

  // When the sync trace stops, the async request should be completely inactive
  auto trace = controller.syncStopTrace();
  EXPECT_NE(trace, nullptr);
  EXPECT_FALSE(controller.isActive());

  // Use asyncStep() being no-op as a proxy to validate that async was cancelled
  for (int i = 0; i < 10; ++i) {
    controller.asyncStep();
    EXPECT_FALSE(controller.isActive());
  }

  const std::string logFile = logUrlToPath(asyncCfg.activitiesLogUrl());
  EXPECT_FALSE(traceFileHasContent(logFile)) << logFile;
}

TEST(ActivityProfilerController, IgnoreAsyncRequestsWhileSyncTraceIsActive) {
  auto traceFile = createTempTraceFile("libkineto_test", ".json");
  Config asyncCfg;
  bool success = asyncCfg.parse(fmt::format(
      R"CFG(
    PROFILE_START_ITERATION = 4
    ACTIVITIES_WARMUP_ITERATIONS = 1
    ACTIVITIES_ITERATIONS = 2
    ACTIVITIES_DURATION_SECS = 1
    ACTIVITIES_LOG_FILE = {}
  )CFG",
      traceFile.path()));
  ASSERT_TRUE(success);

  ActivityProfilerController controller(ConfigLoader::instance(), true);
  controller.asyncStep();

  Config syncCfg;
  syncCfg.setClientDefaults();
  syncCfg.validate(system_clock::now());

  // Start a sync trace
  controller.syncPrepareTrace(syncCfg);
  EXPECT_TRUE(controller.isActive());
  EXPECT_FALSE(controller.canAcceptConfig());

  // Accept an async config while the sync trace is active
  // This should be blocked at the controller level
  controller.acceptConfig(asyncCfg);
  EXPECT_TRUE(controller.isActive());

  controller.syncStartTrace();
  auto trace = controller.syncStopTrace();
  EXPECT_NE(trace, nullptr);
  EXPECT_FALSE(controller.isActive());

  // After sync runs to completion, check that the async request was not
  // accepted.
  // Use asyncStep() being no-op as a proxy to validate that async was cancelled
  for (int i = 0; i < 10; ++i) {
    controller.asyncStep();
    EXPECT_FALSE(controller.isActive());
  }

  const std::string logFile = logUrlToPath(asyncCfg.activitiesLogUrl());
  EXPECT_FALSE(traceFileHasContent(logFile)) << logFile;
}

TEST(ActivityProfilerController, PrepareTracePreemptsActiveAsyncRequest) {
  auto traceFile = createTempTraceFile("libkineto_test", ".json");
  Config asyncCfg;
  bool success = asyncCfg.parse(fmt::format(
      R"CFG(
    PROFILE_START_ITERATION = 3
    ACTIVITIES_WARMUP_ITERATIONS = 1
    ACTIVITIES_ITERATIONS = 4
    ACTIVITIES_DURATION_SECS = 1
    ACTIVITIES_LOG_FILE = {}
  )CFG",
      traceFile.path()));
  ASSERT_TRUE(success);

  ActivityProfilerController controller(ConfigLoader::instance(), true);
  controller.asyncStep();

  // Start an async request, step until it's active
  controller.acceptConfig(asyncCfg);
  EXPECT_FALSE(controller.isActive());
  controller.asyncStep();
  EXPECT_FALSE(controller.isActive());
  controller.asyncStep();
  EXPECT_TRUE(controller.isActive());
  controller.asyncStep();
  EXPECT_TRUE(controller.isActive());

  // Start a sync trace, which should preempt the async request
  Config syncCfg;
  syncCfg.setClientDefaults();
  syncCfg.validate(system_clock::now());
  controller.syncPrepareTrace(syncCfg);
  EXPECT_TRUE(controller.isActive());
  controller.syncStartTrace();
  auto trace = controller.syncStopTrace();
  EXPECT_NE(trace, nullptr);
  EXPECT_FALSE(controller.isActive());

  // Use asyncStep() being no-op as a proxy to validate that async was cancelled
  for (int i = 0; i < 10; ++i) {
    controller.asyncStep();
    EXPECT_FALSE(controller.isActive());
  }

  const std::string logFile = logUrlToPath(asyncCfg.activitiesLogUrl());
  EXPECT_FALSE(traceFileHasContent(logFile)) << logFile;
}

// While an async trace is already active, the controller drops a second
// on-demand request instead of scheduling it. This complements the tests above,
// which cover a sync trace blocking or preempting async.
TEST(ActivityProfilerController, SecondAsyncRequestIgnoredWhileAsyncActive) {
  auto firstFile = createTempTraceFile("libkineto_test_first", ".json");
  auto secondFile = createTempTraceFile("libkineto_test_second", ".json");

  Config firstCfg;
  ASSERT_TRUE(firstCfg.parse(fmt::format(
      R"CFG(
    PROFILE_START_ITERATION = 3
    ACTIVITIES_WARMUP_ITERATIONS = 1
    ACTIVITIES_ITERATIONS = 4
    ACTIVITIES_DURATION_SECS = 1
    ACTIVITIES_LOG_FILE = {}
  )CFG",
      firstFile.path())));

  Config secondCfg;
  ASSERT_TRUE(secondCfg.parse(fmt::format(
      R"CFG(
    PROFILE_START_ITERATION = 3
    ACTIVITIES_WARMUP_ITERATIONS = 1
    ACTIVITIES_ITERATIONS = 4
    ACTIVITIES_DURATION_SECS = 1
    ACTIVITIES_LOG_FILE = {}
  )CFG",
      secondFile.path())));

  ActivityProfilerController controller(ConfigLoader::instance(), true);
  controller.asyncStep();

  // Bring the first request to the active state; acceptConfig reports that it
  // scheduled the request.
  EXPECT_TRUE(controller.acceptConfig(firstCfg));
  controller.asyncStep();
  controller.asyncStep();
  EXPECT_TRUE(controller.isActive());
  EXPECT_FALSE(controller.canAcceptConfig());

  // A second on-demand request arriving now is rejected by the controller.
  EXPECT_FALSE(controller.acceptConfig(secondCfg));
  EXPECT_TRUE(controller.isActive());

  // The rejected request also never produced a trace.
  const std::string secondLog = logUrlToPath(secondCfg.activitiesLogUrl());
  EXPECT_FALSE(traceFileHasContent(secondLog)) << secondLog;
}

// canAcceptConfig() gates on the profiler being idle: it is true before a
// request and false once one becomes active. Return-to-idle happens on the
// background loop after collection finishes and is not asserted here.
TEST(ActivityProfilerController, CanAcceptConfigReflectsActiveState) {
  auto traceFile = createTempTraceFile("libkineto_test", ".json");

  Config cfg;
  ASSERT_TRUE(cfg.parse(fmt::format(
      R"CFG(
    PROFILE_START_ITERATION = 3
    ACTIVITIES_WARMUP_ITERATIONS = 1
    ACTIVITIES_ITERATIONS = 2
    ACTIVITIES_DURATION_SECS = 1
    ACTIVITIES_LOG_FILE = {}
  )CFG",
      traceFile.path())));

  ActivityProfilerController controller(ConfigLoader::instance(), true);
  EXPECT_TRUE(controller.canAcceptConfig());

  controller.asyncStep();
  EXPECT_TRUE(controller.acceptConfig(cfg));
  controller.asyncStep();
  controller.asyncStep();
  EXPECT_TRUE(controller.isActive());
  EXPECT_FALSE(controller.canAcceptConfig());
}
