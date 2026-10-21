/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "include/Config.h"
#include "src/GenericActivityProfiler.h"
#include "src/output_membuf.h"

using namespace KINETO_NAMESPACE;
using namespace std::chrono;

namespace {

class RecordingGpuProfiler : public GenericActivityProfiler {
 public:
  RecordingGpuProfiler() : GenericActivityProfiler(/*cpuOnly=*/false) {}

  std::vector<std::string> events;
  bool useCpuOnly{false};

 protected:
  void ensureGpuTracingReady() override {
    events.emplace_back("ready");
    if (useCpuOnly) {
      cpuOnly_ = true;
    }
  }

  void clearGpuActivities() override {
    events.emplace_back("clear");
  }

  void requestGpuTracingTeardown() override {
    events.emplace_back("teardown");
  }

  void enableGpuTracing() override {
    events.emplace_back("enable");
  }
};

// Run one full synchronous trace and leave the profiler in the post-teardown
// state the guards protect: processTrace() -> finalizeTrace() std::move()s
// traceBuffers_ out (now null). The bare processTrace() does not call
// resetInternal(), so acceptCpuTraces_ stays true. Net: acceptCpuTraces_ ==
// true and traceBuffers_ == nullptr -- the state a late transferCpuTrace() or a
// redundant processTrace() can hit.
void runSyncTraceLeavingStaleState(
    GenericActivityProfiler& profiler,
    const Config& cfg) {
  const auto now = system_clock::now();
  profiler.configure(cfg, now);
  profiler.startTrace(now);
  profiler.stopTrace(now);
  MemoryTraceLogger logger(cfg);
  profiler.processTrace(logger);
}

} // namespace

class GenericActivityProfilerTeardownTest : public ::testing::Test {
 protected:
  void SetUp() override {
    cfg_ = std::make_unique<Config>();
    cfg_->validate(system_clock::now());
  }
  std::unique_ptr<Config> cfg_;
};

// A span arriving after teardown must be discarded, not dereferenced.
// acceptCpuTraces_ is still true here, so the acceptCpuTraces_ gate alone is
// not enough -- transferCpuTrace() must also null-check traceBuffers_. Before
// the fix it ran cpu.push_back() on the null traceBuffers_ and crashed.
TEST_F(GenericActivityProfilerTeardownTest, LateTransferCpuTraceIsDiscarded) {
  GenericActivityProfiler profiler(/*cpuOnly=*/true);
  runSyncTraceLeavingStaleState(profiler, *cfg_);

  auto lateSpan = std::make_unique<CpuTraceBuffer>();
  lateSpan->span = TraceSpan(0, 0, "late span");
  lateSpan->gpuOpCount = 0;
  profiler.transferCpuTrace(std::move(lateSpan)); // must not crash
}

// A redundant processTrace() after teardown must be a no-op. Unlike
// transferCpuTrace(), processTraceInternal() has no acceptCpuTraces_ gate, so
// without the null guard it dereferences the moved-out traceBuffers_
// (cpu.size()).
TEST_F(GenericActivityProfilerTeardownTest, RedundantProcessTraceIsNoOp) {
  GenericActivityProfiler profiler(/*cpuOnly=*/true);
  runSyncTraceLeavingStaleState(profiler, *cfg_);

  MemoryTraceLogger logger(*cfg_);
  profiler.processTrace(logger); // must not crash
}

TEST_F(
    GenericActivityProfilerTeardownTest,
    ConfigureEnsuresGpuTracingReadyWithoutRequestingTeardown) {
  RecordingGpuProfiler profiler;

  profiler.configure(*cfg_, system_clock::now());
  EXPECT_EQ(
      profiler.events, (std::vector<std::string>{"ready", "clear", "enable"}));

  profiler.events.clear();
  profiler.reset();
  EXPECT_EQ(profiler.events, (std::vector<std::string>{"clear", "teardown"}));
}

TEST_F(GenericActivityProfilerTeardownTest, ConfigureContinuesCpuOnly) {
  RecordingGpuProfiler profiler;
  profiler.useCpuOnly = true;

  profiler.configure(*cfg_, system_clock::now());
  EXPECT_EQ(profiler.events, (std::vector<std::string>{"ready"}));

  profiler.events.clear();
  profiler.configure(*cfg_, system_clock::now());
  EXPECT_TRUE(profiler.events.empty());
}
