/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifdef USE_KINETO

#include <gtest/gtest.h>

#include <atomic>
#include <memory>
#include <string>

#include <torch/csrc/autograd/profiler_kineto.h>
#include <torch/csrc/profiler/standalone/privateuse1_profiler.h>

#include <IActivityProfiler.h>
#include <libkineto.h>

namespace {

// Flag to track if Kineto actually called our mock profiler's configure()
// This proves the full flow: registration → Kineto stores it → Kineto uses it
std::atomic<bool> g_configure_called{false};

// Mock IActivityProfilerSession implementation
class MockProfilerSession : public libkineto::IActivityProfilerSession {
 public:
  MockProfilerSession() = default;

  void start() override {
    status_ = libkineto::TraceStatus::RECORDING;
  }

  void stop() override {
    status_ = libkineto::TraceStatus::PROCESSING;
  }

  std::vector<std::string> errors() override {
    return {};
  }

  void processTrace(
      [[maybe_unused]] libkineto::ActivityLogger& logger) override {}

  std::unique_ptr<libkineto::DeviceInfo> getDeviceInfo() override {
    return std::make_unique<libkineto::DeviceInfo>(
        /*id=*/0,
        /*sortIndex=*/0,
        /*name=*/"MockDevice",
        /*label=*/"mock_device");
  }

  std::vector<libkineto::ResourceInfo> getResourceInfos() override {
    return {};
  }

  std::unique_ptr<libkineto::CpuTraceBuffer> getTraceBuffer() override {
    return nullptr;
  }
};

// Mock IActivityProfiler implementation for testing
class MockPrivateUse1Profiler : public libkineto::IActivityProfiler {
 public:
  MockPrivateUse1Profiler() = default;

  const std::string& name() const override {
    return name_;
  }

  const std::set<libkineto::ActivityType>& availableActivities()
      const override {
    return activities_;
  }

  std::unique_ptr<libkineto::IActivityProfilerSession> configure(
      [[maybe_unused]] const std::set<libkineto::ActivityType>& activity_types,
      [[maybe_unused]] const libkineto::Config& config) override {
    g_configure_called = true;
    return std::make_unique<MockProfilerSession>();
  }

  std::unique_ptr<libkineto::IActivityProfilerSession> configure(
      [[maybe_unused]] int64_t ts_ms,
      [[maybe_unused]] int64_t duration_ms,
      const std::set<libkineto::ActivityType>& activity_types,
      const libkineto::Config& config) override {
    return configure(activity_types, config);
  }

 private:
  std::string name_{"mock_privateuse1"};
  std::set<libkineto::ActivityType> activities_{
      libkineto::ActivityType::CPU_OP};
};

} // namespace

// Test that the registry singleton works correctly
TEST(PrivateUse1ProfilerTest, RegistrySingleton) {
  auto& registry1 =
      torch::profiler::impl::PrivateUse1ProfilerRegistry::instance();
  auto& registry2 =
      torch::profiler::impl::PrivateUse1ProfilerRegistry::instance();
  EXPECT_EQ(&registry1, &registry2);
}

// Test that registering a factory works
TEST(PrivateUse1ProfilerTest, RegisterFactory) {
  auto& registry =
      torch::profiler::impl::PrivateUse1ProfilerRegistry::instance();

  registry.registerFactory(
      []() { return std::make_unique<MockPrivateUse1Profiler>(); });

  EXPECT_TRUE(registry.hasFactory());
}

// Test that onKinetoInit triggers registration when factory is registered
TEST(PrivateUse1ProfilerTest, OnKinetoInitForwarding) {
  auto& registry =
      torch::profiler::impl::PrivateUse1ProfilerRegistry::instance();

  // Register factory if not already registered
  if (!registry.hasFactory()) {
    registry.registerFactory(
        []() { return std::make_unique<MockPrivateUse1Profiler>(); });
  }

  registry.onKinetoInit();

  // Verify the factory was forwarded to Kineto
  EXPECT_TRUE(registry.hasFactory());
  EXPECT_TRUE(registry.isRegisteredWithKineto());
}

// End-to-end test: Start a profiling session and verify mock profiler is used
TEST(PrivateUse1ProfilerTest, EndToEndProfiling) {
  using namespace torch::autograd::profiler;
  using namespace torch::profiler::impl;

  // Reset the flag before the test
  g_configure_called = false;

  auto& registry = PrivateUse1ProfilerRegistry::instance();
  if (!registry.hasFactory()) {
    registry.registerFactory(
        []() { return std::make_unique<MockPrivateUse1Profiler>(); });
  }

  // Create profiler config with KINETO_PRIVATEUSE1 state
  ProfilerConfig config(
      ProfilerState::KINETO_PRIVATEUSE1,
      /*report_input_shapes=*/false,
      /*profile_memory=*/false,
      /*with_stack=*/false,
      /*with_flops=*/false,
      /*with_modules=*/false);

  std::set<ActivityType> activities{ActivityType::CPU};

  // This triggers Kineto init and onKinetoInit
  prepareProfiler(config, activities);
  enableProfiler(config, activities, {at::RecordScope::FUNCTION});

  EXPECT_TRUE(registry.isRegisteredWithKineto());

  // Stop profiler
  auto result = disableProfiler();

  EXPECT_TRUE(registry.isRegisteredWithKineto());

  // Verify Kineto-side: Kineto actually called our mock's configure()
  EXPECT_TRUE(g_configure_called);
}

#endif // USE_KINETO
