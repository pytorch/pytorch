/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <chrono>
#include <string>

#include "include/Config.h"
#include "src/GenericActivityProfiler.h"
#include "src/SyncActivityProfilerHandler.h"

using namespace std::chrono;
using namespace KINETO_NAMESPACE;

TEST(SyncActivityProfilerHandler, Cancel) {
  GenericActivityProfiler profiler(/*cpu only*/ true);
  SyncActivityProfilerHandler handler(profiler);

  // Cancel when inactive is a no-op
  EXPECT_FALSE(handler.isSyncActive());
  handler.cancel();
  EXPECT_FALSE(handler.isSyncActive());

  // Cancel after prepareTrace
  Config cfg;
  cfg.validate(system_clock::now());
  handler.prepareTrace(cfg);
  EXPECT_TRUE(handler.isSyncActive());
  handler.cancel();
  EXPECT_FALSE(handler.isSyncActive());
}

// A profiler with GPU support that counts the GPU hooks each trace calls.
class GpuHookCounter : public GenericActivityProfiler {
 public:
  GpuHookCounter() : GenericActivityProfiler(/*cpuOnly=*/false) {}

  int enableCalls{0};
  int disableCalls{0};
  int processCalls{0};

 protected:
  void enableGpuTracing() override {
    ++enableCalls;
  }
  void disableGpuTracing() override {
    ++disableCalls;
  }
  void processGpuActivities(ActivityLogger& /*logger*/) override {
    ++processCalls;
  }
};

TEST(SyncActivityProfilerHandler, GpuTracingFollowsEachTraceConfig) {
  GpuHookCounter profiler;
  SyncActivityProfilerHandler handler(profiler);

  auto runTrace = [&](const std::string& configStr) {
    Config cfg;
    EXPECT_TRUE(cfg.parse(configStr));
    cfg.validate(system_clock::now());
    handler.prepareTrace(cfg);
    handler.startTrace();
    handler.stopTrace();
  };

  runTrace("GPU_TRACING_ENABLED=false");
  EXPECT_EQ(profiler.enableCalls, 0);
  EXPECT_EQ(profiler.disableCalls, 0);
  EXPECT_EQ(profiler.processCalls, 0);

  // GPU tracing is allowed by default.
  runTrace("");
  EXPECT_EQ(profiler.enableCalls, 1);
  EXPECT_EQ(profiler.disableCalls, 1);
  EXPECT_EQ(profiler.processCalls, 1);

  // An earlier GPU trace does not arm a later trace that opts out.
  runTrace("GPU_TRACING_ENABLED=false");
  EXPECT_EQ(profiler.enableCalls, 1);
  EXPECT_EQ(profiler.disableCalls, 1);
  EXPECT_EQ(profiler.processCalls, 1);
}
