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

  bool toggledOn() const {
    return toggleState_.load();
  }

  int bufferSizeCalls{0};
  int enableCalls{0};
  int disableCalls{0};
  int clearCalls{0};
  int processCalls{0};
  int syncCalls{0};
  int resetCalls{0};
  // Stands in for the GPU tracer stopping on its own, e.g. when it runs out
  // of buffers.
  bool gpuStopped{false};

 protected:
  void setMaxGpuBufferSize(int64_t /*size*/) override {
    ++bufferSizeCalls;
  }
  void enableGpuTracing() override {
    ++enableCalls;
  }
  void disableGpuTracing() override {
    ++disableCalls;
  }
  void clearGpuActivities() override {
    ++clearCalls;
  }
  void processGpuActivities(ActivityLogger& /*logger*/) override {
    ++processCalls;
  }
  void synchronizeGpuDevice() override {
    ++syncCalls;
  }
  void onResetTraceData() override {
    ++resetCalls;
  }
  bool isGpuCollectionStopped() const override {
    return gpuStopped;
  }
};

namespace {

void prepareTrace(
    SyncActivityProfilerHandler& handler,
    const std::string& configStr) {
  Config cfg;
  EXPECT_TRUE(cfg.parse(configStr));
  cfg.validate(system_clock::now());
  handler.prepareTrace(cfg);
}

} // namespace

TEST(SyncActivityProfilerHandler, GpuTracingFollowsEachTraceConfig) {
  GpuHookCounter profiler;
  SyncActivityProfilerHandler handler(profiler);

  auto runTrace = [&](const std::string& configStr) {
    prepareTrace(handler, configStr);
    handler.startTrace();
    handler.stopTrace();
  };
  auto expectHookCalls = [&](int traces) {
    EXPECT_EQ(profiler.bufferSizeCalls, traces);
    EXPECT_EQ(profiler.enableCalls, traces);
    EXPECT_EQ(profiler.disableCalls, traces);
    EXPECT_EQ(profiler.processCalls, traces);
    // GPU state is cleared when a trace is configured and again when it is
    // reset.
    EXPECT_EQ(profiler.clearCalls, 2 * traces);
    EXPECT_EQ(profiler.resetCalls, 2 * traces);
  };

  runTrace("DEVICE_TRACING_ENABLED=false");
  expectHookCalls(0);

  // GPU tracing is allowed by default.
  runTrace("");
  expectHookCalls(1);

  // An earlier GPU trace does not arm a later trace that opts out.
  runTrace("DEVICE_TRACING_ENABLED=false");
  expectHookCalls(1);
}

TEST(SyncActivityProfilerHandler, OnlyTracesThatArmGpuReportGpuStop) {
  GpuHookCounter profiler;
  SyncActivityProfilerHandler handler(profiler);

  prepareTrace(handler, "");
  handler.startTrace();
  profiler.gpuStopped = true;
  EXPECT_TRUE(profiler.isStopped());
  handler.stopTrace();

  // The GPU tracer keeps its stopped state, which a trace that never armed it
  // must not report.
  prepareTrace(handler, "DEVICE_TRACING_ENABLED=false");
  handler.startTrace();
  EXPECT_FALSE(profiler.isStopped());
  handler.stopTrace();
}

TEST(SyncActivityProfilerHandler, DynamicToggleFollowsTraceConfig) {
  GpuHookCounter profiler;
  SyncActivityProfilerHandler handler(profiler);

  auto runToggledTrace = [&](const std::string& configStr) {
    prepareTrace(handler, configStr);
    handler.startTrace();
    handler.toggleCollectionDynamic(false);
    handler.toggleCollectionDynamic(true);
    handler.stopTrace();
  };

  // Toggling a trace that opts out never reaches the GPU hooks.
  runToggledTrace("DEVICE_TRACING_ENABLED=false");
  EXPECT_EQ(profiler.enableCalls, 0);
  EXPECT_EQ(profiler.disableCalls, 0);
  EXPECT_EQ(profiler.syncCalls, 0);

  // A GPU trace still toggles. Configure and the toggle each enable, and the
  // toggle and stop each disable.
  runToggledTrace("");
  EXPECT_EQ(profiler.enableCalls, 2);
  EXPECT_EQ(profiler.disableCalls, 2);
  EXPECT_EQ(profiler.syncCalls, 2);
}

TEST(SyncActivityProfilerHandler, ToggleAfterStopDoesNotArmGpu) {
  GpuHookCounter profiler;
  SyncActivityProfilerHandler handler(profiler);

  prepareTrace(handler, "");
  handler.startTrace();
  handler.stopTrace();
  ASSERT_EQ(profiler.enableCalls, 1);
  ASSERT_EQ(profiler.disableCalls, 1);

  // No trace is left to disarm GPU tracing, so a toggle must not arm it.
  handler.toggleCollectionDynamic(true);
  EXPECT_EQ(profiler.enableCalls, 1);
  EXPECT_EQ(profiler.syncCalls, 0);
}

TEST(SyncActivityProfilerHandler, EveryTraceStartsToggledOn) {
  GpuHookCounter profiler;
  SyncActivityProfilerHandler handler(profiler);

  // Stopping a GPU trace toggles collection off.
  prepareTrace(handler, "");
  handler.startTrace();
  handler.stopTrace();
  EXPECT_FALSE(profiler.toggledOn());

  // A trace that opts out of GPU tracing still starts toggled on, so its
  // first toggle off reaches its child sessions.
  prepareTrace(handler, "DEVICE_TRACING_ENABLED=false");
  EXPECT_TRUE(profiler.toggledOn());
  handler.startTrace();
  handler.stopTrace();
}
