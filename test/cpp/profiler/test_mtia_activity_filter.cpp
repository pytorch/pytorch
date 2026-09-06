/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifdef USE_KINETO

#include <gtest/gtest.h>

#include <memory>
#include <set>
#include <string>
#include <utility>

#include <c10/util/Exception.h>
#include <c10/util/env.h>
#include <torch/csrc/profiler/kineto_shim.h>

#include <IActivityProfiler.h>
#include <libkineto.h>

namespace {

constexpr auto kCollectiveProfilingEnv =
    "TORCH_PROFILER_ENABLE_COLLECTIVE_PROFILING";

class CapturingProfiler : public libkineto::IActivityProfiler {
 public:
  explicit CapturingProfiler(
      std::shared_ptr<std::set<libkineto::ActivityType>> configured_activities)
      : configured_activities_{std::move(configured_activities)} {}

  const std::string& name() const override {
    return name_;
  }
  const std::set<libkineto::ActivityType>& availableActivities()
      const override {
    return activities_;
  }
  std::unique_ptr<libkineto::IActivityProfilerSession> configure(
      const std::set<libkineto::ActivityType>& activity_types,
      [[maybe_unused]] const libkineto::Config& config) override {
    *configured_activities_ = activity_types;
    return nullptr;
  }
  std::unique_ptr<libkineto::IActivityProfilerSession> configure(
      [[maybe_unused]] int64_t ts_ms,
      [[maybe_unused]] int64_t duration_ms,
      const std::set<libkineto::ActivityType>& activity_types,
      const libkineto::Config& config) override {
    return configure(activity_types, config);
  }

 private:
  std::string name_{"capturing"};
  std::set<libkineto::ActivityType> activities_{
      libkineto::ActivityType::MTIA_INSIGHT,
      libkineto::ActivityType::COLLECTIVE_COMM};
  std::shared_ptr<std::set<libkineto::ActivityType>> configured_activities_;
};

} // namespace

TEST(MtiaActivityFilterTest, CollectiveSelectionUsesFiltersAndAvailability) {
  using torch::autograd::profiler::ActivityType;
  using torch::profiler::impl::ExperimentalConfig;
  using torch::profiler::impl::kineto::ActivityFilter;
  using torch::profiler::impl::kineto::ActivitySet;
  using torch::profiler::impl::kineto::prepareTrace;

  const auto configured_activities =
      std::make_shared<std::set<libkineto::ActivityType>>();
  // Libkineto has no unregister API. This dedicated binary contains one test,
  // so the process-lifetime factory cannot affect another test case.
  libkineto::api().registerProfilerFactory([configured_activities]() {
    return std::make_unique<CapturingProfiler>(configured_activities);
  });

  const auto prepareAndCapture = [&](const ActivitySet& activities,
                                     const ActivityFilter& activity_filter,
                                     const ExperimentalConfig& config) {
    configured_activities->clear();
    // Keep the test hardware-independent while exercising the same activity
    // selection logic passed to registered profilers.
    prepareTrace(
        /*cpuOnly=*/true,
        activities,
        config,
        /*trace_id=*/"",
        activity_filter);
  };

  const std::set<libkineto::ActivityType> default_cpu_activities{
      libkineto::ActivityType::CPU_OP,
      libkineto::ActivityType::CPU_INSTANT_EVENT,
      libkineto::ActivityType::USER_ANNOTATION,
      libkineto::ActivityType::EXTERNAL_CORRELATION,
      libkineto::ActivityType::XPU_RUNTIME,
      libkineto::ActivityType::XPU_DRIVER,
      libkineto::ActivityType::CUDA_RUNTIME,
      libkineto::ActivityType::CUDA_DRIVER,
      libkineto::ActivityType::PYTHON_FUNCTION,
      libkineto::ActivityType::PRIVATEUSE1_RUNTIME,
      libkineto::ActivityType::PRIVATEUSE1_DRIVER,
  };

  c10::utils::set_env(kCollectiveProfilingEnv, "1");

  prepareAndCapture(
      {ActivityType::CPU, ActivityType::MTIA},
      {{ActivityType::MTIA, {"MTIA_INSIGHT"}}},
      ExperimentalConfig{});
  auto expected_cpu_and_insight = default_cpu_activities;
  expected_cpu_and_insight.insert(libkineto::ActivityType::MTIA_INSIGHT);
  EXPECT_EQ(*configured_activities, expected_cpu_and_insight);

  prepareAndCapture(
      {ActivityType::MTIA},
      {{ActivityType::MTIA, {"COLLECTIVE_COMM"}}},
      ExperimentalConfig{});
  const std::set<libkineto::ActivityType> collective_only{
      libkineto::ActivityType::COLLECTIVE_COMM};
  EXPECT_EQ(*configured_activities, collective_only);

  prepareAndCapture({ActivityType::CPU}, {}, ExperimentalConfig{});
  auto expected_cpu_and_collectives = default_cpu_activities;
  expected_cpu_and_collectives.insert(libkineto::ActivityType::COLLECTIVE_COMM);
  EXPECT_EQ(*configured_activities, expected_cpu_and_collectives);

  ExperimentalConfig custom_config;
  custom_config.custom_profiler_config = "disable_runtime_events";
  prepareAndCapture({ActivityType::MTIA}, {}, custom_config);
  const std::set<libkineto::ActivityType> expected_custom_config{
      libkineto::ActivityType::MTIA_CCP_EVENTS,
      libkineto::ActivityType::MTIA_INSIGHT,
      libkineto::ActivityType::MTIA_COUNTERS,
      libkineto::ActivityType::COLLECTIVE_COMM,
  };
  EXPECT_EQ(*configured_activities, expected_custom_config);

  EXPECT_THROW(
      prepareAndCapture(
          {ActivityType::MTIA},
          {{ActivityType::MTIA, {"MTIA_INSIGHT"}}},
          custom_config),
      c10::Error);

  c10::utils::set_env(kCollectiveProfilingEnv, "0");
#if !defined(KINETO_HAS_HCCL_PROFILER)
  prepareAndCapture(
      {ActivityType::MTIA},
      {{ActivityType::MTIA, {"MTIA_INSIGHT", "COLLECTIVE_COMM"}}},
      ExperimentalConfig{});
  const std::set<libkineto::ActivityType> insight_only{
      libkineto::ActivityType::MTIA_INSIGHT};
  EXPECT_EQ(*configured_activities, insight_only);
#endif
}

#endif // USE_KINETO
