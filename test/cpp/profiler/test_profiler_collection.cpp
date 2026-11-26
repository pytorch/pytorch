/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Tests for torch/csrc/profiler/collection.cpp. The PrivateUse1 profiler
// machinery is used here only as a vehicle for injecting synthetic kineto
// activities into a real profiling session.
//
// To add new tests in this file:
//   - Register child profilers via libkineto::api().registerProfilerFactory()
//     directly. PrivateUse1ProfilerRegistry::registerFactory() is one-shot
//     (it forwards to libkineto on first call only), so it cannot layer
//     additional mocks on top of whatever a prior test already installed.
//   - Wrap assertions on TORCH_WARN_ONCE output in c10::WarningUtils::
//     WarnAlways(true) to defeat per-process rate limiting.
//   - Capture warnings with CapturingWarningHandler under a
//     WarningHandlerGuard for isolated capture.
//   - Assert what your test PRODUCES, not what is ABSENT. Activities from
//     prior tests' child profilers will still flow through setParents.
//
// If a future test genuinely needs a pristine Kineto state (e.g., asserts a
// negative, or relies on TORCH_WARN_ONCE rate limiting being in effect),
// split it out in .ci/pytorch/test.sh with a -k filter — same pattern that
// isolates EndToEndProfiling in test_privateuse1_profiler.

#ifdef USE_KINETO

#include <gtest/gtest.h>

#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <c10/util/Exception.h>
#include <torch/csrc/autograd/profiler_kineto.h>
#include <torch/csrc/profiler/collection.h>
#include <torch/csrc/profiler/standalone/privateuse1_profiler.h>

#include <ActivityTraceInterface.h>
#include <GenericTraceActivity.h>
#include <IActivityProfiler.h>
#include <libkineto.h>
#include <output_base.h>

namespace {

// Mock session that emits two CUDA_RUNTIME activities sharing the same
// async-CPU-GPU flow start ID. Drives the duplicate-flow path in
// TransferEvents::setParents().
class DuplicateFlowMockSession : public libkineto::IActivityProfilerSession {
 public:
  DuplicateFlowMockSession() {
    span_ = libkineto::TraceSpan{0, 0, "duplicate_flow_span"};
    for (int i = 0; i < 2; ++i) {
      libkineto::GenericTraceActivity a(
          span_,
          libkineto::ActivityType::CUDA_RUNTIME,
          "duplicate_flow_launch");
      a.startTime = 100 + i;
      a.endTime = 200 + i;
      a.id = i; // distinct correlation IDs
      a.flow.id = 42; // identical flow ID across both activities
      a.flow.type = libkineto::kLinkAsyncCpuGpu;
      a.flow.start = 1; // both are flow starts
      activities_.push_back(std::move(a));
    }
  }

  void start() override {
    status_ = libkineto::TraceStatus::RECORDING;
  }
  void stop() override {
    status_ = libkineto::TraceStatus::PROCESSING;
  }
  std::vector<std::string> errors() override {
    return {};
  }

  // Push the duplicate-flow activities into the logger so they reach the
  // MemoryTraceLogger's activities_ list, which is what TransferEvents
  // consumes via ActivityTrace::activities().
  void processTrace(libkineto::ActivityLogger& logger) override {
    for (const auto& a : activities_) {
      logger.handleGenericActivity(a);
    }
  }

  std::unique_ptr<libkineto::DeviceInfo> getDeviceInfo() override {
    return std::make_unique<libkineto::DeviceInfo>(
        /*id=*/1,
        /*sortIndex=*/1,
        /*name=*/"DuplicateFlowMockDevice",
        /*label=*/"duplicate_flow_mock_device");
  }

  std::vector<libkineto::ResourceInfo> getResourceInfos() override {
    return {};
  }

  std::unique_ptr<libkineto::CpuTraceBuffer> getTraceBuffer() override {
    return nullptr;
  }

 private:
  libkineto::TraceSpan span_{0, 0, ""};
  std::vector<libkineto::GenericTraceActivity> activities_;
};

class DuplicateFlowMockProfiler : public libkineto::IActivityProfiler {
 public:
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
    return std::make_unique<DuplicateFlowMockSession>();
  }

  std::unique_ptr<libkineto::IActivityProfilerSession> configure(
      [[maybe_unused]] int64_t ts_ms,
      [[maybe_unused]] int64_t duration_ms,
      const std::set<libkineto::ActivityType>& activity_types,
      const libkineto::Config& config) override {
    return configure(activity_types, config);
  }

 private:
  std::string name_{"duplicate_flow_mock"};
  std::set<libkineto::ActivityType> activities_{
      libkineto::ActivityType::CUDA_RUNTIME};
};

// Minimal IActivityProfiler used to ensure PrivateUse1ProfilerRegistry has
// a factory so libkineto initializes. Emits no activities.
class NoopPrivateUse1Profiler : public libkineto::IActivityProfiler {
 public:
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
    class Session : public libkineto::IActivityProfilerSession {
     public:
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
        return nullptr;
      }
      std::vector<libkineto::ResourceInfo> getResourceInfos() override {
        return {};
      }
      std::unique_ptr<libkineto::CpuTraceBuffer> getTraceBuffer() override {
        return nullptr;
      }
    };
    return std::make_unique<Session>();
  }
  std::unique_ptr<libkineto::IActivityProfilerSession> configure(
      [[maybe_unused]] int64_t ts_ms,
      [[maybe_unused]] int64_t duration_ms,
      const std::set<libkineto::ActivityType>& activity_types,
      const libkineto::Config& config) override {
    return configure(activity_types, config);
  }

 private:
  std::string name_{"noop_privateuse1"};
  std::set<libkineto::ActivityType> activities_{
      libkineto::ActivityType::CPU_OP};
};

class CapturingWarningHandler : public c10::WarningHandler {
 public:
  void process(const c10::Warning& warning) override {
    messages_.push_back(warning.msg());
  }
  const std::vector<std::string>& messages() const {
    return messages_;
  }

 private:
  std::vector<std::string> messages_;
};

} // namespace

// A backend producing duplicate async-CPU-GPU flow start IDs must be handled
// gracefully.
TEST(ProfilerCollectionTest, DuplicateFlowIdHandledGracefully) {
  using namespace torch::autograd::profiler;
  using namespace torch::profiler::impl;

  // Capture warnings so the new diagnostic can be asserted. TORCH_WARN_ONCE
  // is rate-limited per process; force every WARN_ONCE to behave like WARN
  // for the duration of this test.
  CapturingWarningHandler handler;
  c10::WarningUtils::WarningHandlerGuard handler_guard(&handler);
  c10::WarningUtils::WarnAlways warn_always_guard(true);

  // Register a child profiler directly with libkineto. The session it
  // returns is what injects duplicate flow IDs into the trace.
  libkineto::api().registerProfilerFactory(
      []() { return std::make_unique<DuplicateFlowMockProfiler>(); });

  // The PrivateUse1 registry needs a factory so prepareProfiler triggers
  // libkineto init, which in turn materializes the factory registered above.
  if (!PrivateUse1ProfilerRegistry::instance().hasFactory()) {
    PrivateUse1ProfilerRegistry::instance().registerFactory(
        []() { return std::make_unique<NoopPrivateUse1Profiler>(); });
  }

  ProfilerConfig config(
      ProfilerState::KINETO_PRIVATEUSE1,
      /*report_input_shapes=*/false,
      /*profile_memory=*/false,
      /*with_stack=*/false,
      /*with_flops=*/false,
      /*with_modules=*/false);

  std::set<ActivityType> activities{ActivityType::CPU};

  prepareProfiler(config, activities);
  enableProfiler(config, activities, {at::RecordScope::FUNCTION});
  auto result = disableProfiler();
  EXPECT_NE(result, nullptr);

  bool saw_per_id_warning = false;
  bool saw_summary_warning = false;
  for (const auto& msg : handler.messages()) {
    if (msg.find("Profiler produced duplicate flow start") !=
        std::string::npos) {
      saw_per_id_warning = true;
    }
    if (msg.find("duplicate flow start ID") != std::string::npos &&
        msg.find("Profiler observed") != std::string::npos) {
      saw_summary_warning = true;
    }
  }
  EXPECT_TRUE(saw_per_id_warning);
  EXPECT_TRUE(saw_summary_warning);
}

namespace {
using torch::profiler::impl::EventType;
using torch::profiler::impl::ExtraFields;
using torch::profiler::impl::PyFrameState;
using torch::profiler::impl::Result;

std::shared_ptr<Result> makePyCall(
    int64_t start_ns,
    int64_t end_ns,
    size_t python_tid) {
  PyFrameState frame{0, at::StringView("f.py"), at::StringView("fn")};
  PyFrameState caller{0, at::StringView("caller.py"), at::StringView("caller")};
  ExtraFields<EventType::PyCall>::args_t args{
      frame, std::nullopt, std::nullopt};
  return Result::create(
      start_ns,
      /*start_tid=*/7,
      torch::profiler::impl::kineto::DeviceAndResource(),
      ExtraFields<EventType::PyCall>(
          end_ns, python_tid, caller, std::move(args)));
}

std::shared_ptr<Result> makePyCCall(
    int64_t start_ns,
    int64_t end_ns,
    size_t python_tid) {
  PyFrameState caller{0, at::StringView("caller.py"), at::StringView("caller")};
  return Result::create(
      start_ns,
      /*start_tid=*/7,
      torch::profiler::impl::kineto::DeviceAndResource(),
      ExtraFields<EventType::PyCCall>(
          end_ns, python_tid, caller, at::StringView("<built-in method foo>")));
}

class LatePythonEventTracer
    : public torch::profiler::impl::python_tracer::PythonTracerBase {
 public:
  void stop() override {}
  void restart() override {}
  void register_gc_callback() override {}
  std::vector<std::shared_ptr<Result>> getEvents(
      std::function<c10::time_t(c10::approx_time_t)> /*time_converter*/,
      std::vector<
          torch::profiler::impl::python_tracer::CompressedEvent>& /*enters*/,
      c10::time_t end_time_ns) override {
    EXPECT_EQ(end_time_ns, 200);
    return {
        makePyCall(110, 150, 1),
        makePyCCall(120, 160, 1),
        makePyCall(200, 200, 1),
        makePyCCall(200, 200, 1),
        makePyCall(201, 250, 1),
        makePyCCall(201, 250, 1)};
  }
};

class PythonEventTraceEndTest : public ::testing::Test {
 protected:
  void SetUp() override {
    torch::profiler::impl::python_tracer::registerTracer(
        [](torch::profiler::impl::RecordQueue*)
            -> std::unique_ptr<
                torch::profiler::impl::python_tracer::PythonTracerBase> {
          return std::make_unique<LatePythonEventTracer>();
        });
  }
  void TearDown() override {
    torch::profiler::impl::python_tracer::registerTracer(nullptr);
  }
};

} // namespace

TEST_F(PythonEventTraceEndTest, DropsLateStartsAfterPythonEventsAreAppended) {
  using namespace torch::autograd::profiler;
  using namespace torch::profiler::impl;
  for (const bool trace_only : {false, true}) {
    SCOPED_TRACE(trace_only);
    ProfilerConfig config(ProfilerState::KINETO);
    config.with_stack = true;
    config.experimental_config.trace_only = trace_only;
    kineto::prepareTrace({ActivityType::CPU}, config.experimental_config);
    libkineto::api().activityProfiler().startTrace();
    RecordQueue queue(config, {ActivityType::CPU});
    auto [records, trace] = queue.getRecords(
        [](c10::approx_time_t t) { return static_cast<c10::time_t>(t); },
        /*start_time_ns=*/100,
        /*end_time_ns=*/200);

    std::vector<int64_t> starts;
    for (const auto& record : records) {
      starts.push_back(record->start_time_ns_);
    }
    const std::vector<int64_t> expected_starts{110, 120, 200, 200};
    EXPECT_EQ(starts, expected_starts);
    ASSERT_TRUE(trace);
    ASSERT_TRUE(trace->get());
    const auto* activities = trace->get()->activities();
    ASSERT_NE(activities, nullptr);
    std::vector<int64_t> trace_starts;
    for (const auto* activity : *activities) {
      if (activity->type() == libkineto::ActivityType::PYTHON_FUNCTION) {
        trace_starts.push_back(activity->timestamp());
        EXPECT_GE(activity->duration(), 0);
      }
    }
    EXPECT_EQ(trace_starts, expected_starts);
  }
}

TEST(PythonEventTimestampClampTest, ClampsOverrunsWithinPythonThread) {
  auto parent = makePyCall(100, 500, /*python_tid=*/1);
  auto valid_child = makePyCall(110, 180, /*python_tid=*/1);
  auto overrun = makePyCCall(200, 10000, /*python_tid=*/1);
  auto descendant = makePyCall(220, 10000, /*python_tid=*/1);
  auto other_thread = makePyCall(250, 900, /*python_tid=*/2);
  auto adjacent = makePyCall(500, 600, /*python_tid=*/1);
  std::vector<std::shared_ptr<Result>> events{
      parent, valid_child, overrun, descendant, other_thread, adjacent};

  torch::profiler::impl::python_tracer::clampOverrunningPythonEvents(events);

  const std::vector<int64_t> expected_end_times{500, 180, 500, 500, 900, 600};
  const std::vector<int64_t> actual_end_times{
      parent->endTimeNS(),
      valid_child->endTimeNS(),
      overrun->endTimeNS(),
      descendant->endTimeNS(),
      other_thread->endTimeNS(),
      adjacent->endTimeNS()};
  EXPECT_EQ(actual_end_times, expected_end_times);
}

TEST(ProfilerCollectionTest, CapsFinishedEventsAtTraceEnd) {
  using namespace torch::autograd::profiler;
  using namespace torch::profiler::impl;
  for (const bool trace_only : {false, true}) {
    SCOPED_TRACE(trace_only);
    ProfilerConfig config(ProfilerState::KINETO);
    config.experimental_config.trace_only = trace_only;
    config.experimental_config.disable_external_correlation = true;
    kineto::prepareTrace({ActivityType::CPU}, config.experimental_config);
    libkineto::api().activityProfiler().startTrace();
    RecordQueue queue(config, {ActivityType::CPU});
    auto record = [&](const char* name, int64_t start, int64_t end) {
      at::RecordFunction fn(at::RecordScope::FUNCTION);
      fn.before(name);
      auto context = queue.getSubqueue()->begin_op(fn);
      context->event_->start_time_ = start;
      context->event_->end_time_ = end;
    };
    record("window_valid", 110, 150);
    record("window_late", 120, 250);
    record("window_zero", 200, 200);
    record("window_after", 201, 250);
    record(
        "window_unfinished",
        130,
        std::numeric_limits<c10::approx_time_t>::min());

    auto [records, trace] = queue.getRecords(
        [](c10::approx_time_t t) { return static_cast<c10::time_t>(t); },
        /*start_time_ns=*/100,
        /*end_time_ns=*/200);
    ASSERT_TRUE(trace);
    ASSERT_TRUE(trace->get());
    const auto* activities = trace->get()->activities();
    ASSERT_NE(activities, nullptr);
    std::vector<std::string> names;
    std::vector<int64_t> ends;
    std::vector<std::string> finished;
    bool saw_unfinished = false;
    for (const auto* activity : *activities) {
      if (activity->name() == "window_unfinished") {
        EXPECT_EQ(activity->getMetadataValue("finished"), "false");
        saw_unfinished = true;
      } else if (activity->name().starts_with("window_")) {
        names.push_back(activity->name());
        ends.push_back(activity->timestamp() + activity->duration());
        finished.push_back(activity->getMetadataValue("finished"));
      }
    }
    const std::vector<std::string> expected_names{
        "window_valid", "window_late", "window_zero"};
    const std::vector<int64_t> expected_ends{150, 200, 200};
    const std::vector<std::string> expected_finished{"", "", ""};
    EXPECT_EQ(names, expected_names);
    EXPECT_EQ(ends, expected_ends);
    EXPECT_EQ(finished, expected_finished);
    EXPECT_TRUE(saw_unfinished);
  }
}

#endif // USE_KINETO
