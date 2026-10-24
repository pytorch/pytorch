/*
 * Copyright (C) Intel Corporation 
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

#include <c10/core/DeviceType.h>
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
      libkineto::ActivityType::XPU_SYNC};
  std::shared_ptr<std::set<libkineto::ActivityType>> configured_activities_;
};

} // namespace

TEST(XpuActivityFilterTest, SyncSelectionFollowsDefaultsAndFilters) {
  using torch::autograd::profiler::ActivityType;
  using torch::profiler::impl::ExperimentalConfig;
  using torch::profiler::impl::kineto::ActivityFilter;
  using torch::profiler::impl::kineto::ActivitySet;
  using torch::profiler::impl::kineto::prepareTrace;

  const auto configured_activities =
      std::make_shared<std::set<libkineto::ActivityType>>();
  // Libkineto has no unregister API, so this factory lives for the process.
  // The other test here never calls prepareTrace(), so it is unaffected.
  libkineto::api().registerProfilerFactory([configured_activities]() {
    return std::make_unique<CapturingProfiler>(configured_activities);
  });

  // A true collectivesProfilerExists() adds COLLECTIVE_COMM to every case
  // below. Pinned so expectations do not depend on the environment; the
  // assertions test membership rather than whole sets so they also hold
  // under KINETO_HAS_HCCL_PROFILER.
  c10::utils::set_env(kCollectiveProfilingEnv, "0");

  const auto prepareAndCapture = [&](const ActivitySet& activities,
                                     const ActivityFilter& activity_filter) {
    configured_activities->clear();
    prepareTrace(
        /*cpuOnly=*/true,
        activities,
        ExperimentalConfig{},
        /*trace_id=*/"",
        activity_filter);
    return *configured_activities;
  };

  // A bare ProfilerActivity.XPU collects the group defaults, now with sync.
  const auto defaults =
      prepareAndCapture({ActivityType::CPU, ActivityType::XPU}, {});
  EXPECT_TRUE(defaults.contains(libkineto::ActivityType::XPU_SYNC));

  // Naming XPU_SYNC selects it and nothing else. Before it joined the XPU
  // group the name was unrecognised and became an XPUPTI_PROFILER_METRICS
  // entry, which enabled XPU_SCOPE_PROFILER plus CONCURRENT_KERNEL.
  const auto sync_only = prepareAndCapture(
      {ActivityType::XPU}, {{ActivityType::XPU, {"XPU_SYNC"}}});
  EXPECT_TRUE(sync_only.contains(libkineto::ActivityType::XPU_SYNC));
  EXPECT_FALSE(sync_only.contains(libkineto::ActivityType::XPU_SCOPE_PROFILER));
  EXPECT_FALSE(sync_only.contains(libkineto::ActivityType::CONCURRENT_KERNEL));
  EXPECT_FALSE(sync_only.contains(libkineto::ActivityType::XPU_RUNTIME));

  // An activity name mixed with a metric name still splits: the activity is
  // collected, the metric adds the scope profiler and the kernel view.
  const auto sync_and_metrics = prepareAndCapture(
      {ActivityType::XPU}, {{ActivityType::XPU, {"XPU_SYNC", "GpuTime"}}});
  EXPECT_TRUE(sync_and_metrics.contains(libkineto::ActivityType::XPU_SYNC));
  EXPECT_TRUE(
      sync_and_metrics.contains(libkineto::ActivityType::XPU_SCOPE_PROFILER));
  EXPECT_TRUE(
      sync_and_metrics.contains(libkineto::ActivityType::CONCURRENT_KERNEL));

  // An empty dict still opts out of the whole XPU group.
  const auto nothing =
      prepareAndCapture({ActivityType::XPU}, {{ActivityType::XPU, {}}});
  EXPECT_FALSE(nothing.contains(libkineto::ActivityType::XPU_SYNC));

  // It needs a PTI view, so it is in the XPU group only and deliberately not
  // in the CPU group.
  const auto cpu_only = prepareAndCapture({ActivityType::CPU}, {});
  EXPECT_FALSE(cpu_only.contains(libkineto::ActivityType::XPU_SYNC));

  // Unrecognised names become metric names, which PyTorch cannot validate,
  // so a misspelling collects no synchronization events.
  const auto typo = prepareAndCapture(
      {ActivityType::XPU}, {{ActivityType::XPU, {"XPU_SYNCH"}}});
  EXPECT_FALSE(typo.contains(libkineto::ActivityType::XPU_SYNC));
}

TEST(XpuActivityFilterTest, SyncActivityIsReportedOnTheHost) {
  using torch::autograd::profiler::deviceTypeFromActivity;

  // XPUPTI reports these with device -1 and resource = thread id, GPU
  // barriers included, so XPU would inflate self_xpu_time_total. CPU is also
  // what XPU_RUNTIME and XPU_DRIVER return.
  EXPECT_EQ(
      deviceTypeFromActivity(libkineto::ActivityType::XPU_SYNC),
      c10::DeviceType::CPU);
}

#endif // USE_KINETO
