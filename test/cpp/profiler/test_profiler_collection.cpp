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

#include <memory>
#include <string>
#include <vector>

#include <c10/util/Exception.h>
#include <torch/csrc/autograd/profiler_kineto.h>
#include <torch/csrc/profiler/collection.h>
#include <torch/csrc/profiler/standalone/privateuse1_profiler.h>

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

std::shared_ptr<Result> makePyCall(int64_t start_ns, int64_t end_ns, uint64_t tid) {
  PyFrameState frame{0, at::StringView("f.py"), at::StringView("fn")};
  PyFrameState caller{0, at::StringView("caller.py"), at::StringView("caller")};
  ExtraFields<EventType::PyCall>::args_t args{frame, std::nullopt, std::nullopt};
  return Result::create(
      start_ns,
      tid,
      torch::profiler::impl::kineto::DeviceAndResource(),
      ExtraFields<EventType::PyCall>(end_ns, tid, caller, std::move(args)));
}

std::shared_ptr<Result> makePyCCall(int64_t start_ns, int64_t end_ns, uint64_t tid) {
  PyFrameState caller{0, at::StringView("caller.py"), at::StringView("caller")};
  return Result::create(
      start_ns,
      tid,
      torch::profiler::impl::kineto::DeviceAndResource(),
      ExtraFields<EventType::PyCCall>(
          end_ns, tid, caller, at::StringView("<built-in method foo>")));
}

// A cpu_op (RecordFunction) frame, used as a non-Python frame nested between two
// Python frames (mirrors the autograd engine's evaluate_function frame).
std::shared_ptr<Result> makeTorchOp(int64_t start_ns, int64_t end_ns, uint64_t tid) {
  torch::profiler::impl::TorchOpBasicFields basic;
  basic.name_ = "autograd::engine::evaluate_function";
  return Result::create(
      start_ns,
      tid,
      torch::profiler::impl::kineto::DeviceAndResource(),
      ExtraFields<EventType::TorchOp>(
          std::move(basic),
          /*correlation_id=*/0,
          /*end_time_ns=*/end_ns,
          /*inputs=*/{},
          /*concrete_inputs=*/{},
          /*jit_stack=*/{},
          /*jit_modules=*/{},
          /*extra_args=*/{},
          /*extra_meta=*/{},
          /*kwinputs=*/{},
          /*device_fallback=*/torch::profiler::impl::FallbackPair{},
          /*allow_tf32_cublas=*/false,
          /*perf_event_counters=*/nullptr));
}

} // namespace

// These tests exercise clampOverrunningPythonEvents directly on hand-built trees
// (parent_ links wired manually), so they neither depend on nor require exposing
// the internal build_tree pass.

// A Python C-call whose return event was dropped has its end clamped past its
// caller. The clamp must bring it back to the caller's end (a callee cannot
// outlive its caller).
TEST(ProfilerTreeClampTest, OrphanChildClampedToParentEnd) {
  const uint64_t tid = 1;
  auto parent = makePyCall(/*start=*/100, /*end=*/200, tid);
  auto child = makePyCCall(/*start=*/150, /*end=*/10000, tid);
  child->parent_ = parent;
  std::vector<std::shared_ptr<Result>> events{parent, child};

  torch::profiler::impl::clampOverrunningPythonEvents(events);

  EXPECT_EQ(child->endTimeNS(), 200);
  EXPECT_EQ(parent->endTimeNS(), 200);
}

// Frames genuinely on the stack when profiling stops are all clamped to the
// SAME profiler-end, so child end == parent end. The clamp must leave these
// untouched (they are not orphans).
TEST(ProfilerTreeClampTest, OnStackAtStopNotClamped) {
  const uint64_t tid = 1;
  auto parent = makePyCall(/*start=*/100, /*end=*/10000, tid);
  auto child = makePyCCall(/*start=*/150, /*end=*/10000, tid);
  child->parent_ = parent;
  std::vector<std::shared_ptr<Result>> events{parent, child};

  torch::profiler::impl::clampOverrunningPythonEvents(events);

  EXPECT_EQ(child->endTimeNS(), 10000);
  EXPECT_EQ(parent->endTimeNS(), 10000);
}

// The orphan's nearest Python ancestor is separated from it by a non-Python
// cpu_op frame (as with autograd's evaluate_function between run_backward and
// apply). The clamp must walk past the cpu_op to the Python ancestor.
TEST(ProfilerTreeClampTest, OrphanClampedThroughNonPythonParent) {
  const uint64_t tid = 1;
  auto py_ancestor = makePyCCall(/*start=*/100, /*end=*/200, tid);
  auto cpu_op = makeTorchOp(/*start=*/110, /*end=*/190, tid);
  auto orphan = makePyCall(/*start=*/120, /*end=*/10000, tid);
  orphan->parent_ = cpu_op;
  cpu_op->parent_ = py_ancestor;
  std::vector<std::shared_ptr<Result>> events{py_ancestor, cpu_op, orphan};

  torch::profiler::impl::clampOverrunningPythonEvents(events);

  // orphan's nearest Python ancestor is the PyCCall (end 200), reached by
  // walking past the cpu_op.
  EXPECT_EQ(orphan->endTimeNS(), 200);
  EXPECT_EQ(py_ancestor->endTimeNS(), 200);
}

// A short event whose Python-ancestor link is stale -- it starts AFTER that
// ancestor already returned (as seen with a mis-attributed `apply` on an
// autograd worker) -- is not a duration bug. The `ancestor_end >= event_start`
// guard must leave it untouched; clamping would wrongly force end < start. The
// parent link is set directly to model the mis-attribution regardless of how it
// arose during tree building.
TEST(ProfilerTreeClampTest, StaleAncestorAttributionNotClamped) {
  const uint64_t tid = 1;
  auto ancestor = makePyCCall(/*start=*/100, /*end=*/200, tid);
  auto child = makePyCall(/*start=*/500, /*end=*/600, tid);
  child->parent_ = ancestor;
  std::vector<std::shared_ptr<Result>> events{ancestor, child};

  torch::profiler::impl::clampOverrunningPythonEvents(events);

  EXPECT_EQ(child->endTimeNS(), 600);
}

// A child whose nearest Python ancestor is on a DIFFERENT Python thread is not a
// real caller/callee (build_tree can pair unrelated threads when start_tid_ is
// unresolved). The python_tid guard must leave it untouched even though it is
// nested and overruns.
TEST(ProfilerTreeClampTest, DifferentPythonThreadNotClamped) {
  auto ancestor = makePyCCall(/*start=*/100, /*end=*/5000, /*tid=*/1);
  auto child = makePyCall(/*start=*/500, /*end=*/10000, /*tid=*/2);
  child->parent_ = ancestor;
  std::vector<std::shared_ptr<Result>> events{ancestor, child};

  torch::profiler::impl::clampOverrunningPythonEvents(events);

  EXPECT_EQ(child->endTimeNS(), 10000);
}

#endif // USE_KINETO
