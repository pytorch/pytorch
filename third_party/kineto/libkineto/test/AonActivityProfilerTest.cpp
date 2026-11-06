/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "include/Config.h"
#include "src/AonActivityProfiler.h"

using namespace KINETO_NAMESPACE;

namespace {

struct FakeConsumerState {
  int initializeCalls{0};
  int startCalls{0};
  int stopCalls{0};
  bool initializeResult{true};
  bool startResult{true};
  bool stopResult{true};
  std::filesystem::path initializedPath;
};

class FakeAonTraceConsumer final : public IAonTraceConsumer {
 public:
  explicit FakeAonTraceConsumer(FakeConsumerState& state) : state_(state) {}

  bool initialize(const std::filesystem::path& databasePath) override {
    ++state_.initializeCalls;
    state_.initializedPath = databasePath;
    return state_.initializeResult;
  }

  bool start() override {
    ++state_.startCalls;
    return state_.startResult;
  }

  bool stop() override {
    ++state_.stopCalls;
    return state_.stopResult;
  }

 private:
  FakeConsumerState& state_;
};

class ScopedGpuProfilerMode {
 public:
  explicit ScopedGpuProfilerMode(const char* value) {
    const char* oldValue = std::getenv("KINETO_GPU_PROFILER");
    if (oldValue != nullptr) {
      oldValue_ = oldValue;
      hadOldValue_ = true;
    }
    if (value == nullptr) {
      unsetenv("KINETO_GPU_PROFILER");
    } else {
      setenv("KINETO_GPU_PROFILER", value, 1);
    }
  }

  ~ScopedGpuProfilerMode() {
    if (hadOldValue_) {
      setenv("KINETO_GPU_PROFILER", oldValue_.c_str(), 1);
    } else {
      unsetenv("KINETO_GPU_PROFILER");
    }
  }

 private:
  std::string oldValue_;
  bool hadOldValue_{false};
};

} // namespace

TEST(AonActivityProfilerTest, RegistrationSeamAdvertisesNoActivitiesYet) {
  AonActivityProfiler profiler;
  libkineto::Config config;

  EXPECT_EQ(profiler.name(), "NVIDIA Nsight Perf (AON)");
  EXPECT_TRUE(profiler.availableActivities().empty());
  EXPECT_EQ(profiler.configure({}, config), nullptr);
  EXPECT_EQ(profiler.configure(0, 1, {}, config), nullptr);
}

TEST(AonActivityProfilerTest, AonModeRequiresExplicitValidSelection) {
  {
    ScopedGpuProfilerMode mode(nullptr);
    EXPECT_FALSE(isAonGpuProfilerSelected());
  }
  {
    ScopedGpuProfilerMode mode("cupti");
    EXPECT_FALSE(isAonGpuProfilerSelected());
  }
  {
    ScopedGpuProfilerMode mode("aon");
    EXPECT_TRUE(isAonGpuProfilerSelected());
  }
  {
    ScopedGpuProfilerMode mode("invalid");
    EXPECT_FALSE(isAonGpuProfilerSelected());
  }
}

TEST(AonActivitySessionTest, DrivesConsumerLifecycleOnce) {
  FakeConsumerState state;
  AonActivitySession session(
      std::make_unique<FakeAonTraceConsumer>(state),
      std::filesystem::path{"unused"},
      false);

  ASSERT_TRUE(session.prepare());
  EXPECT_TRUE(session.prepare());
  EXPECT_EQ(state.initializeCalls, 1);

  session.start();
  session.start();
  EXPECT_EQ(state.startCalls, 1);
  EXPECT_EQ(session.status(), libkineto::TraceStatus::RECORDING);

  session.stop();
  session.stop();
  EXPECT_EQ(state.stopCalls, 1);
  EXPECT_EQ(session.status(), libkineto::TraceStatus::PROCESSING);
  EXPECT_TRUE(session.errors().empty());
}

TEST(AonActivitySessionTest, ReportsExternalServiceFailure) {
  FakeConsumerState state;
  state.initializeResult = false;
  AonActivitySession session(
      std::make_unique<FakeAonTraceConsumer>(state),
      std::filesystem::path{"unused"},
      false);

  EXPECT_FALSE(session.prepare());
  EXPECT_EQ(session.status(), libkineto::TraceStatus::ERROR);
  ASSERT_EQ(session.errors().size(), 1);
  EXPECT_NE(session.errors()[0].find("external nvidia-nsight-perf"),
            std::string::npos);

  session.start();
  session.stop();
  EXPECT_EQ(state.startCalls, 0);
  EXPECT_EQ(state.stopCalls, 0);
}

TEST(AonActivitySessionTest, CleansTemporaryDatabase) {
  FakeConsumerState state;
  const auto databasePath = std::filesystem::temp_directory_path() /
      "kineto-aon-session-cleanup-test";
  std::filesystem::remove_all(databasePath);
  std::filesystem::create_directories(databasePath);

  {
    AonActivitySession session(
        std::make_unique<FakeAonTraceConsumer>(state), databasePath);
    ASSERT_TRUE(session.prepare());
    EXPECT_TRUE(std::filesystem::exists(databasePath));
  }

  EXPECT_FALSE(std::filesystem::exists(databasePath));
}
