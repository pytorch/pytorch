/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <strings.h>
#include <time.h>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <optional>

#include <unistd.h>

#ifdef __linux__
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#endif

#include "include/Config.h"
#include "include/MetadataFieldCatalog.h"
#include "include/libkineto.h"
#include "include/output_base.h"
#include "include/time_since_epoch.h"
#include "src/ActivityTrace.h"
#include "src/RocmActivityProfiler.h"
#include "src/RocmStreamQueue.h"

#include "src/RocprofActivity.h"
#include "src/RocprofActivityApi.h"
#include "src/RocprofLogger.h"

#include "src/output_json.h"
#include "src/output_membuf.h"

#include "src/Logger.h"
#include "src/ThrowUtil.h"
#include "test/MockActivitySubProfiler.h"
#include "test/MockCpuActivityBuffer.h"
#include "test/TestUtils.h"

using namespace std::chrono;
using namespace KINETO_NAMESPACE;
using namespace libkineto::test;

// API ID macros for rocprofiler-sdk
#define HIP_LAUNCH_KERNEL ROCPROFILER_HIP_RUNTIME_API_ID_hipLaunchKernel
#define HIP_MEMCPY ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpy
#define HIP_MEMSET ROCPROFILER_HIP_RUNTIME_API_ID_hipMemsetAsync
#define HIP_MALLOC ROCPROFILER_HIP_RUNTIME_API_ID_hipMalloc
#define HIP_FREE ROCPROFILER_HIP_RUNTIME_API_ID_hipFree
#define RUNTIME_DOMAIN ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API

namespace {
bool isAsyncCopy(const rocprofAsyncRow& async) {
  return async.domain == ROCPROFILER_BUFFER_TRACING_MEMORY_COPY;
}

bool isAsyncKernel(const rocprofAsyncRow& async) {
  return async.domain == ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH;
}

struct RocmStreamTypedMetadataVisitor final : public ITypedMetadataVisitor {
  void visitValue(const MetadataField<int64_t>& field, int64_t value) override {
    if (field.name == RocmMetadataFields::kStream.name) {
      stream = value;
    }
  }

  void visitValue(const MetadataField<uint64_t>& field, uint64_t value)
      override {
    if (field.name == RocmMetadataFields::kHsaQueue.name) {
      hsaQueue = value;
    }
  }

  void visitValue(
      [[maybe_unused]] const MetadataField<double>& field,
      [[maybe_unused]] double value) override {}
  void visitValue(
      [[maybe_unused]] const MetadataField<bool>& field,
      [[maybe_unused]] bool value) override {}
  void visitValue(
      [[maybe_unused]] const MetadataField<std::string>& field,
      [[maybe_unused]] std::string_view value) override {}
  void visitValue(
      [[maybe_unused]] const MetadataField<std::vector<int64_t>>& field,
      [[maybe_unused]] const std::vector<int64_t>& value) override {}
  void visitValue(
      [[maybe_unused]] const MetadataField<std::vector<std::string>>& field,
      [[maybe_unused]] const std::vector<std::string>& value) override {}
  void visitValue(
      [[maybe_unused]] const MetadataField<RawJson>& field,
      [[maybe_unused]] const RawJson& value) override {}
  void visitValue(
      [[maybe_unused]] const MetadataField<InputShapes>& field,
      [[maybe_unused]] const InputShapes& value) override {}

  void visitUnsupported(std::string_view /*name*/) override {}

  void beginDict(std::string_view /*name*/) override {}
  void endDict() override {}

  std::optional<int64_t> stream;
  std::optional<uint64_t> hsaQueue;
};
} // namespace

// Provides ability to easily create test ROCm ops using the shared types
// from RocLogger.h (rocprofKernelRow, rocprofAsyncRow, etc.)
struct MockRocLogger {
  void addCorrelationActivity(
      uint64_t correlation,
      RocLogger::CorrelationDomain domain,
      uint64_t externalId) {
    externalCorrelations_[domain].emplace_back(correlation, externalId);
  }

  void addRuntimeKernelActivity(
      uint32_t cid,
      int64_t start_ns,
      int64_t end_ns,
      int64_t correlation,
      uint64_t stream = 0) {
    rocprofKernelRow* row = new rocprofKernelRow(
        correlation,
        RUNTIME_DOMAIN,
        cid,
        processId(),
        systemThreadId(),
        start_ns,
        end_ns,
        nullptr,
        nullptr,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        reinterpret_cast<hipStream_t>(stream));
    activities_.push_back(row);
  }

  void addRuntimeMallocActivity(
      uint32_t cid,
      int64_t start_ns,
      int64_t end_ns,
      int64_t correlation) {
    rocprofMallocRow* row = new rocprofMallocRow(
        correlation,
        RUNTIME_DOMAIN,
        cid,
        processId(),
        systemThreadId(),
        start_ns,
        end_ns,
        nullptr,
        1);
    activities_.push_back(row);
  }

  void addRuntimeMemsetActivity(
      uint32_t cid,
      int64_t start_ns,
      int64_t end_ns,
      int64_t correlation,
      size_t size,
      uint64_t stream = 0) {
    rocprofMemsetRow* row = new rocprofMemsetRow(
        correlation,
        RUNTIME_DOMAIN,
        cid,
        processId(),
        systemThreadId(),
        start_ns,
        end_ns,
        size,
        reinterpret_cast<hipStream_t>(stream));
    activities_.push_back(row);
  }

  void addRuntimeCopyActivity(
      uint32_t cid,
      int64_t start_ns,
      int64_t end_ns,
      int64_t correlation,
      hipMemcpyKind kind = hipMemcpyHostToHost,
      uint64_t stream = 0) {
    rocprofCopyRow* row = new rocprofCopyRow(
        correlation,
        RUNTIME_DOMAIN,
        cid,
        processId(),
        systemThreadId(),
        start_ns,
        end_ns,
        nullptr,
        nullptr,
        1,
        kind,
        reinterpret_cast<hipStream_t>(stream));
    activities_.push_back(row);
  }

  void addKernelActivity(
      int64_t start_ns,
      int64_t end_ns,
      int64_t correlation,
      uint64_t queue = 1) {
    rocprofAsyncRow* row = new rocprofAsyncRow(
        correlation,
        ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH,
        0,
        0,
        0,
        queue,
        start_ns,
        end_ns,
        std::string("kernel"));
    activities_.push_back(row);
  }

  void addMemcpyH2DActivity(
      int64_t start_ns,
      int64_t end_ns,
      int64_t correlation,
      uint64_t queue = 2) {
    rocprofAsyncRow* row = new rocprofAsyncRow(
        correlation,
        ROCPROFILER_BUFFER_TRACING_MEMORY_COPY,
        0,
        ROCPROFILER_MEMORY_COPY_HOST_TO_DEVICE,
        0,
        queue,
        start_ns,
        end_ns,
        std::string());
    activities_.push_back(row);
  }

  void addMemcpyD2HActivity(
      int64_t start_ns,
      int64_t end_ns,
      int64_t correlation) {
    rocprofAsyncRow* row = new rocprofAsyncRow(
        correlation,
        ROCPROFILER_BUFFER_TRACING_MEMORY_COPY,
        0,
        ROCPROFILER_MEMORY_COPY_DEVICE_TO_HOST,
        0,
        2,
        start_ns,
        end_ns,
        std::string());
    activities_.push_back(row);
  }

  ~MockRocLogger() {
    while (!activities_.empty()) {
      auto act = activities_.back();
      activities_.pop_back();
      free(act);
    }
  }

  std::vector<rocprofBase*> activities_;
  std::vector<std::pair<uint64_t, uint64_t>>
      externalCorrelations_[RocLogger::CorrelationDomain::size];
};

// Mock parts of the ActivityApi
class MockRocActivities : public RocprofActivityApi {
 public:
  bool isAvailable() const override {
    ++availabilityChecks;
    return available;
  }

  virtual int processActivities(
      std::function<void(const rocprofBase*)> handler,
      std::function<void(uint64_t, uint64_t, RocLogger::CorrelationDomain)>
          correlationHandler) override {
    ++processActivitiesCalls;
    int count = 0;
    for (int it = RocLogger::CorrelationDomain::begin;
         it < RocLogger::CorrelationDomain::end;
         ++it) {
      auto& externalCorrelations = activityLogger->externalCorrelations_[it];
      for (auto& item : externalCorrelations) {
        correlationHandler(
            item.first,
            item.second,
            static_cast<RocLogger::CorrelationDomain>(it));
      }
      externalCorrelations.clear();
    }
    // Mirror RocprofActivityApi::processActivities: classify the fill/copy
    // dispatches issued for memset/memcpy calls, backfill the streams of the
    // logged GPU rows, then apply the type-based filter before logging.
    detail::classifyRuntimeInternalDispatches(activityLogger->activities_);
    if (isLogged(ActivityType::GPU_MEMCPY) ||
        isLogged(ActivityType::GPU_MEMSET) ||
        isLogged(ActivityType::CONCURRENT_KERNEL)) {
      detail::backfillAsyncStreams(
          activityLogger->activities_, [this](const rocprofAsyncRow& async) {
            return (isAsyncCopy(async) || isAsyncKernel(async)) &&
                detail::isAsyncRowLogged(async, activityMask);
          });
    }
    for (auto& item : activityLogger->activities_) {
      bool filtered = false;
      if (item->type != ROCTRACER_ACTIVITY_ASYNC &&
          !isLogged(ActivityType::CUDA_RUNTIME)) {
        filtered = true;
      } else {
        switch (item->domain) {
          case ROCPROFILER_BUFFER_TRACING_MEMORY_COPY:
            if (!isLogged(ActivityType::GPU_MEMCPY)) {
              filtered = true;
            }
            break;
          case ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH:
          default:
            if (item->type == ROCTRACER_ACTIVITY_ASYNC) {
              filtered = !detail::isAsyncRowLogged(
                  *reinterpret_cast<rocprofAsyncRow*>(item), activityMask);
            } else if (!isLogged(ActivityType::CONCURRENT_KERNEL)) {
              filtered = true;
            }
            break;
        }
      }
      if (!filtered) {
        handler(item);
        ++count;
      }
    }
    return count;
  }

  bool isLogged(ActivityType atype) const {
    return activityMask & (1u << static_cast<uint32_t>(atype));
  }

  std::unique_ptr<MockRocLogger> activityLogger;
  bool available{true};
  mutable int availabilityChecks{0};
  int processActivitiesCalls{0};
  // Types logged by the default config the fixture uses; tests that want a
  // narrower set overwrite this.
  uint32_t activityMask{
      (1u << static_cast<uint32_t>(ActivityType::CUDA_RUNTIME)) |
      (1u << static_cast<uint32_t>(ActivityType::CONCURRENT_KERNEL)) |
      (1u << static_cast<uint32_t>(ActivityType::GPU_MEMCPY)) |
      (1u << static_cast<uint32_t>(ActivityType::GPU_MEMSET))};
};

// Common setup / teardown and helper functions
class RocmActivityProfilerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    profiler_ = std::make_unique<RocmActivityProfiler>(
        rocActivities_, /*cpu only*/ false);
    cfg_ = std::make_unique<Config>();
    cfg_->validate(std::chrono::system_clock::now());
    loggerFactory.addProtocol("file", [](const std::string& url) {
      return std::unique_ptr<ActivityLogger>(new ChromeTraceLogger(url));
    });
  }

  std::unique_ptr<Config> cfg_;
  MockRocActivities rocActivities_;
  std::unique_ptr<RocmActivityProfiler> profiler_;
  ActivityLoggerFactory loggerFactory;
};

TEST_F(RocmActivityProfilerTest, DefersAndCachesGpuAvailabilityProbe) {
  rocActivities_.available = false;
  EXPECT_EQ(rocActivities_.availabilityChecks, 0);

  const auto startTime = std::chrono::system_clock::now();
  profiler_->configure(*cfg_, startTime);
  EXPECT_EQ(rocActivities_.availabilityChecks, 1);

  profiler_->toggleCollectionDynamic(true);
  EXPECT_EQ(rocActivities_.availabilityChecks, 1);
}

TEST_F(RocmActivityProfilerTest, UnavailableRocmFallsBackToCpuOnly) {
  rocActivities_.available = false;
  rocActivities_.activityLogger = std::make_unique<MockRocLogger>();
  const auto startTime = std::chrono::system_clock::now();
  constexpr auto duration = std::chrono::nanoseconds(300);

  profiler_->configure(*cfg_, startTime);
  profiler_->startTrace(startTime);

  const auto startTimeNs = libkineto::timeSinceEpoch(startTime);
  auto cpuOps = std::make_unique<MockCpuActivityBuffer>(
      startTimeNs, startTimeNs + duration.count());
  cpuOps->addOp("cpu_op", startTimeNs + 20, startTimeNs + 50, 1);
  profiler_->transferCpuTrace(std::move(cpuOps));
  profiler_->stopTrace(startTime + duration);

  auto logger = std::make_unique<MemoryTraceLogger>(*cfg_);
  profiler_->processTrace(*logger);

  ActivityTrace trace(std::move(logger), loggerFactory);
  EXPECT_EQ(rocActivities_.availabilityChecks, 1);
  EXPECT_EQ(rocActivities_.processActivitiesCalls, 0);
  ASSERT_EQ(trace.activities()->size(), 1);
  EXPECT_EQ(trace.activities()->front()->name(), "cpu_op");
  EXPECT_EQ(trace.activities()->front()->type(), ActivityType::CPU_OP);
}

TEST_F(RocmActivityProfilerTest, SyncTrace) {
  // Verbose logging is useful for debugging
  std::vector<std::string> log_modules({"RocmActivityProfiler.cpp"});
  SET_LOG_VERBOSITY_LEVEL(2, log_modules);

  // Start and stop profiling
  RocmActivityProfiler profiler(rocActivities_, /*cpu only*/ false);
  int64_t start_time_ns =
      libkineto::timeSinceEpoch(std::chrono::system_clock::now());
  int64_t duration_ns = 300;
  auto start_time = timePointFromNs(start_time_ns);
  profiler.configure(*cfg_, start_time);
  profiler.startTrace(start_time);
  profiler.stopTrace(timePointFromNs(start_time_ns + duration_ns));

  profiler.recordThreadInfo();

  // Log some cpu ops
  auto cpuOps = std::make_unique<MockCpuActivityBuffer>(
      start_time_ns, start_time_ns + duration_ns);
  cpuOps->addOp("op1", start_time_ns + 20, start_time_ns + 50, 1);
  cpuOps->addOp("op2", start_time_ns + 30, start_time_ns + 40, 2);
  cpuOps->addOp("op3", start_time_ns + 100, start_time_ns + 150, 3);
  cpuOps->addOp("op4", start_time_ns + 160, start_time_ns + 180, 4);
  cpuOps->addOp("op5", start_time_ns + 190, start_time_ns + 210, 4);
  profiler.transferCpuTrace(std::move(cpuOps));

  // And some CPU runtime ops, and GPU ops
  auto gpuOps = std::make_unique<MockRocLogger>();
  gpuOps->addRuntimeKernelActivity(
      HIP_LAUNCH_KERNEL, start_time_ns + 33, start_time_ns + 38, 1);
  gpuOps->addRuntimeCopyActivity(
      HIP_MEMCPY, start_time_ns + 110, start_time_ns + 120, 2);
  gpuOps->addRuntimeKernelActivity(
      HIP_LAUNCH_KERNEL, start_time_ns + 130, start_time_ns + 145, 3);
  gpuOps->addRuntimeCopyActivity(
      HIP_MEMCPY, start_time_ns + 165, start_time_ns + 175, 4);
  gpuOps->addRuntimeKernelActivity(
      HIP_LAUNCH_KERNEL, start_time_ns + 195, start_time_ns + 205, 5);
  gpuOps->addKernelActivity(start_time_ns + 50, start_time_ns + 70, 1);
  gpuOps->addMemcpyH2DActivity(start_time_ns + 140, start_time_ns + 150, 2);
  gpuOps->addKernelActivity(start_time_ns + 160, start_time_ns + 220, 3);
  gpuOps->addMemcpyD2HActivity(start_time_ns + 230, start_time_ns + 250, 4);
  gpuOps->addKernelActivity(start_time_ns + 260, start_time_ns + 280, 5);
  rocActivities_.activityLogger = std::move(gpuOps);

  // Have the profiler process them
  auto logger = std::make_unique<MemoryTraceLogger>(*cfg_);
  profiler.processTrace(*logger);

  // Profiler can be reset at this point - logger owns the activities
  profiler.reset();

  // Wrapper that allows iterating over the activities
  ActivityTrace trace(std::move(logger), loggerFactory);
  EXPECT_EQ(trace.activities()->size(), 15);
  std::map<std::string, int> activityCounts;
  std::map<int64_t, int> resourceIds;
  for (auto& activity : *trace.activities()) {
    activityCounts[activity->name()]++;
    resourceIds[activity->resourceId()]++;
    LOG(INFO) << "[test]" << activity->name() << "," << activity->resourceId();
  }
  for (const auto& p : activityCounts) {
    LOG(INFO) << p.first << ": " << p.second;
  }
  // Check all activities are present and names are correct.
  EXPECT_EQ(activityCounts["op1"], 1);
  EXPECT_EQ(activityCounts["op2"], 1);
  EXPECT_EQ(activityCounts["op3"], 1);
  EXPECT_EQ(activityCounts["op4"], 1);
  EXPECT_EQ(activityCounts["op5"], 1);
  EXPECT_EQ(activityCounts["hipLaunchKernel"], 3);
  EXPECT_EQ(activityCounts["Memcpy HtoD (Host -> Device)"], 1);
  EXPECT_EQ(activityCounts["Memcpy DtoH (Device -> Host)"], 1);
  EXPECT_EQ(activityCounts["kernel"], 3);

  auto sysTid = systemThreadId();
  // Check ops and runtime events are on thread sysTid
  EXPECT_EQ(resourceIds[sysTid], 10);
  // Kernels are on stream 1, memcpy on stream 2
  EXPECT_EQ(resourceIds[1], 3);
  EXPECT_EQ(resourceIds[2], 2);

#ifdef __linux__
  auto tmpTrace = createTempTraceFile("libkineto_test", ".json");
  trace.save(tmpTrace.path());
  checkTracefile(tmpTrace.c_str());
#endif
}

TEST_F(RocmActivityProfilerTest, GpuTypedMetadataMatchesLegacyStreamMetadata) {
  RocmActivityProfiler profiler(rocActivities_, /*cpu only*/ false);
  int64_t start_time_ns =
      libkineto::timeSinceEpoch(std::chrono::system_clock::now());
  int64_t duration_ns = 300;
  auto start_time = timePointFromNs(start_time_ns);
  profiler.configure(*cfg_, start_time);
  profiler.startTrace(start_time);
  profiler.stopTrace(timePointFromNs(start_time_ns + duration_ns));

  auto gpuOps = std::make_unique<MockRocLogger>();
  gpuOps->addMemcpyH2DActivity(start_time_ns + 10, start_time_ns + 20, 1, 42);
  rocActivities_.activityLogger = std::move(gpuOps);

  auto logger = std::make_unique<MemoryTraceLogger>(*cfg_);
  profiler.processTrace(*logger);
  profiler.reset();

  ActivityTrace trace(std::move(logger), loggerFactory);
  const ITraceActivity* memcpyActivity = nullptr;
  for (const auto& activity : *trace.activities()) {
    if (activity->name() == "Memcpy HtoD (Host -> Device)") {
      memcpyActivity = activity;
      break;
    }
  }

  ASSERT_NE(memcpyActivity, nullptr);
  EXPECT_EQ(memcpyActivity->resourceId(), 42);

  RocmStreamTypedMetadataVisitor typedMetadata;
  memcpyActivity->visitTypedMetadata(typedMetadata);
  const auto jsonMetadata =
      nlohmann::json::parse("{" + memcpyActivity->metadataJson() + "}");
  EXPECT_EQ(typedMetadata.stream, jsonMetadata["stream"].get<int64_t>());
  EXPECT_EQ(typedMetadata.hsaQueue, jsonMetadata["hsa_queue"].get<uint64_t>());
}

TEST_F(
    RocmActivityProfilerTest,
    HtoDMemcpyUsesRuntimeStreamWhenAsyncQueueIsZero) {
  RocmActivityProfiler profiler(rocActivities_, /*cpu only*/ false);
  int64_t start_time_ns =
      libkineto::timeSinceEpoch(std::chrono::system_clock::now());
  int64_t duration_ns = 300;
  auto start_time = timePointFromNs(start_time_ns);
  profiler.configure(*cfg_, start_time);
  profiler.startTrace(start_time);
  profiler.stopTrace(timePointFromNs(start_time_ns + duration_ns));

  auto gpuOps = std::make_unique<MockRocLogger>();
  gpuOps->addRuntimeKernelActivity(
      HIP_LAUNCH_KERNEL, start_time_ns + 10, start_time_ns + 15, 2, 7);
  gpuOps->addKernelActivity(start_time_ns + 16, start_time_ns + 19, 2, 23);
  gpuOps->addRuntimeCopyActivity(
      HIP_MEMCPY,
      start_time_ns + 20,
      start_time_ns + 30,
      1,
      hipMemcpyHostToDevice,
      7);
  gpuOps->addMemcpyH2DActivity(start_time_ns + 40, start_time_ns + 50, 1, 0);
  rocActivities_.activityLogger = std::move(gpuOps);

  auto logger = std::make_unique<MemoryTraceLogger>(*cfg_);
  profiler.processTrace(*logger);
  profiler.reset();

  ActivityTrace trace(std::move(logger), loggerFactory);
  const ITraceActivity* memcpyActivity = nullptr;
  for (const auto& activity : *trace.activities()) {
    if (activity->name() == "Memcpy HtoD (Host -> Device)") {
      memcpyActivity = activity;
      break;
    }
  }

  ASSERT_NE(memcpyActivity, nullptr);
  // The copy shares HIP stream 7 with the kernel (via runtime correlation), so
  // both remap to the same dense per-device stream index (1).
  EXPECT_EQ(memcpyActivity->resourceId(), 1);

#ifdef __linux__
  auto tmpTrace = createTempTraceFile("libkineto_test", ".json");
  trace.save(tmpTrace.path());

  std::ifstream file(tmpTrace.path());
  ASSERT_TRUE(file.is_open());
  std::string jsonStr(
      (std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  nlohmann::json jsonData = nlohmann::json::parse(jsonStr);

  bool foundMemcpy = false;
  for (const auto& event : jsonData["traceEvents"]) {
    if (event.value("name", "") == "Memcpy HtoD (Host -> Device)") {
      foundMemcpy = true;
      EXPECT_EQ(event["args"]["stream"].get<int64_t>(), 1);
    }
  }
  EXPECT_TRUE(foundMemcpy);
#endif
}

TEST_F(RocmActivityProfilerTest, HtoDMemcpyPrefersRuntimeStreamOverAsyncQueue) {
  RocmActivityProfiler profiler(rocActivities_, /*cpu only*/ false);
  int64_t start_time_ns =
      libkineto::timeSinceEpoch(std::chrono::system_clock::now());
  int64_t duration_ns = 300;
  auto start_time = timePointFromNs(start_time_ns);
  profiler.configure(*cfg_, start_time);
  profiler.startTrace(start_time);
  profiler.stopTrace(timePointFromNs(start_time_ns + duration_ns));

  auto gpuOps = std::make_unique<MockRocLogger>();
  gpuOps->addRuntimeKernelActivity(
      HIP_LAUNCH_KERNEL, start_time_ns + 10, start_time_ns + 15, 2, 7);
  gpuOps->addKernelActivity(start_time_ns + 16, start_time_ns + 19, 2, 23);
  gpuOps->addRuntimeCopyActivity(
      HIP_MEMCPY,
      start_time_ns + 20,
      start_time_ns + 30,
      1,
      hipMemcpyHostToDevice,
      7);
  gpuOps->addMemcpyH2DActivity(start_time_ns + 40, start_time_ns + 50, 1, 42);
  rocActivities_.activityLogger = std::move(gpuOps);

  auto logger = std::make_unique<MemoryTraceLogger>(*cfg_);
  profiler.processTrace(*logger);
  profiler.reset();

  ActivityTrace trace(std::move(logger), loggerFactory);
  const ITraceActivity* memcpyActivity = nullptr;
  for (const auto& activity : *trace.activities()) {
    if (activity->name() == "Memcpy HtoD (Host -> Device)") {
      memcpyActivity = activity;
      break;
    }
  }

  ASSERT_NE(memcpyActivity, nullptr);
  // Even though the async copy carries a nonzero HW queue (42), it is grouped
  // by its real HIP stream (7) -- shared with the kernel -> dense index 1.
  EXPECT_EQ(memcpyActivity->resourceId(), 1);
  // The raw HSA queue (42) is still logged in the event metadata for debugging,
  // even though track placement uses the stream.
  EXPECT_NE(
      memcpyActivity->metadataJson().find("\"hsa_queue\": 42"),
      std::string::npos);
}

TEST_F(
    RocmActivityProfilerTest,
    HtoDMemcpyJoinsRuntimeStreamDespiteQueueAmbiguity) {
  RocmActivityProfiler profiler(rocActivities_, /*cpu only*/ false);
  int64_t start_time_ns =
      libkineto::timeSinceEpoch(std::chrono::system_clock::now());
  int64_t duration_ns = 300;
  auto start_time = timePointFromNs(start_time_ns);
  profiler.configure(*cfg_, start_time);
  profiler.startTrace(start_time);
  profiler.stopTrace(timePointFromNs(start_time_ns + duration_ns));

  auto gpuOps = std::make_unique<MockRocLogger>();
  gpuOps->addRuntimeKernelActivity(
      HIP_LAUNCH_KERNEL, start_time_ns + 10, start_time_ns + 15, 2, 7);
  gpuOps->addKernelActivity(start_time_ns + 16, start_time_ns + 19, 2, 23);
  gpuOps->addRuntimeKernelActivity(
      HIP_LAUNCH_KERNEL, start_time_ns + 20, start_time_ns + 25, 3, 7);
  gpuOps->addKernelActivity(start_time_ns + 26, start_time_ns + 29, 3, 24);
  gpuOps->addRuntimeCopyActivity(
      HIP_MEMCPY,
      start_time_ns + 30,
      start_time_ns + 40,
      1,
      hipMemcpyHostToDevice,
      7);
  gpuOps->addMemcpyH2DActivity(start_time_ns + 50, start_time_ns + 60, 1, 0);
  rocActivities_.activityLogger = std::move(gpuOps);

  auto logger = std::make_unique<MemoryTraceLogger>(*cfg_);
  profiler.processTrace(*logger);
  profiler.reset();

  ActivityTrace trace(std::move(logger), loggerFactory);
  const ITraceActivity* memcpyActivity = nullptr;
  for (const auto& activity : *trace.activities()) {
    if (activity->name() == "Memcpy HtoD (Host -> Device)") {
      memcpyActivity = activity;
      break;
    }
  }

  ASSERT_NE(memcpyActivity, nullptr);
  // The copy's stream is resolved directly from its runtime correlation (HIP
  // stream 7), so HW-queue ambiguity no longer matters -- it joins the shared
  // stream index 1 instead of being dropped to 0.
  EXPECT_EQ(memcpyActivity->resourceId(), 1);
}

// Sends a hipMemsetAsync call and the fill dispatch it triggers through the
// full profiler pipeline (processTrace -> handleRocprofActivity) and checks
// the exported type, bytes and stream.
TEST_F(RocmActivityProfilerTest, RuntimeMemsetIsExportedAsGpuMemset) {
  RocmActivityProfiler profiler(rocActivities_, /*cpu only*/ false);
  int64_t start_time_ns =
      libkineto::timeSinceEpoch(std::chrono::system_clock::now());
  int64_t duration_ns = 300;
  auto start_time = timePointFromNs(start_time_ns);
  profiler.configure(*cfg_, start_time);
  profiler.startTrace(start_time);
  profiler.stopTrace(timePointFromNs(start_time_ns + duration_ns));

  auto gpuOps = std::make_unique<MockRocLogger>();
  // The fill dispatch (correlation 1) shares its correlation id with the
  // hipMemsetAsync call on HIP stream 7; correlation 2 is a user kernel on
  // stream 8.
  gpuOps->addRuntimeMemsetActivity(
      HIP_MEMSET, start_time_ns + 10, start_time_ns + 15, 1, 64, 7);
  gpuOps->addKernelActivity(start_time_ns + 16, start_time_ns + 19, 1, 0);
  gpuOps->addRuntimeKernelActivity(
      HIP_LAUNCH_KERNEL, start_time_ns + 20, start_time_ns + 25, 2, 8);
  gpuOps->addKernelActivity(start_time_ns + 26, start_time_ns + 29, 2, 0);
  rocActivities_.activityLogger = std::move(gpuOps);

  auto logger = std::make_unique<MemoryTraceLogger>(*cfg_);
  profiler.processTrace(*logger);
  profiler.reset();

  ActivityTrace trace(std::move(logger), loggerFactory);
  const ITraceActivity* memsetCall = nullptr;
  const ITraceActivity* memsetEvent = nullptr;
  const ITraceActivity* kernelEvent = nullptr;
  for (const auto& activity : *trace.activities()) {
    if (activity->type() == ActivityType::GPU_MEMSET) {
      memsetEvent = activity;
    } else if (activity->type() == ActivityType::CONCURRENT_KERNEL) {
      kernelEvent = activity;
    } else if (activity->name() == "hipMemsetAsync") {
      memsetCall = activity;
    }
  }

  // The runtime event carries the call's byte count.
  ASSERT_NE(memsetCall, nullptr);
  const auto callMd =
      nlohmann::json::parse("{" + memsetCall->metadataJson() + "}");
  EXPECT_EQ(callMd["bytes"].get<uint64_t>(), 64u);

  // The fill dispatch is exported as a gpu_memset taking the call's bytes
  // and stream (stream 7 remaps to dense index 1, stream 8 to index 2).
  ASSERT_NE(memsetEvent, nullptr);
  EXPECT_EQ(memsetEvent->name(), "Memset (Device)");
  const auto eventMd =
      nlohmann::json::parse("{" + memsetEvent->metadataJson() + "}");
  EXPECT_EQ(eventMd["bytes"].get<uint64_t>(), 64u);
  EXPECT_EQ(memsetEvent->resourceId(), 1);

  ASSERT_NE(kernelEvent, nullptr);
  EXPECT_EQ(kernelEvent->resourceId(), 2);
}

TEST_F(RocmActivityProfilerTest, GpuNCCLCollectiveTest) {
  // Set logging level for debugging purpose
  std::vector<std::string> log_modules(
      {"RocmActivityProfiler.cpp", "output_json.cpp"});
  SET_LOG_VERBOSITY_LEVEL(2, log_modules);

  // Start and stop profiling
  RocmActivityProfiler profiler(rocActivities_, /*cpu only*/ false);
  int64_t start_time_ns =
      libkineto::timeSinceEpoch(std::chrono::system_clock::now());
  int64_t duration_ns = 300;
  auto start_time = timePointFromNs(start_time_ns);
  profiler.configure(*cfg_, start_time);
  profiler.startTrace(start_time);
  profiler.stopTrace(timePointFromNs(start_time_ns + duration_ns));

  int64_t kernelLaunchTime = start_time_ns + 20;
  profiler.recordThreadInfo();

  // Prepare metadata map
  std::unordered_map<std::string, std::string> metadataMap;
  metadataMap.emplace(
      kCollectiveName, fmt::format("\"{}\"", "_allgather_base"));
  metadataMap.emplace(kDtype, fmt::format("\"{}\"", "Float"));
  metadataMap.emplace(kInMsgNelems, "65664");
  metadataMap.emplace(kOutMsgNelems, "131328");
  metadataMap.emplace(kGroupSize, "2");
  metadataMap.emplace(kProcessGroupName, fmt::format("\"{}\"", "12341234"));
  metadataMap.emplace(kProcessGroupDesc, fmt::format("\"{}\"", "test_purpose"));

  std::vector<int64_t> inSplitSizes(50, 0);
  std::string inSplitSizesStr = "";
  // Logic is copied from: https://fburl.com/code/811a3wq8
  if (!inSplitSizes.empty() && inSplitSizes.size() <= kTruncatLength) {
    inSplitSizesStr = fmt::format("\"[{}]\"", fmt::join(inSplitSizes, ", "));
    metadataMap.emplace(kInSplit, inSplitSizesStr);
  } else if (inSplitSizes.size() > kTruncatLength) {
    inSplitSizesStr = fmt::format(
        "\"[{}, ...]\"",
        fmt::join(
            inSplitSizes.begin(), inSplitSizes.begin() + kTruncatLength, ", "));
    metadataMap.emplace(kInSplit, inSplitSizesStr);
  }

  std::vector<int64_t> outSplitSizes(20, 1);
  std::string outSplitSizesStr = "";
  // Logic is copied from: https://fburl.com/code/811a3wq8
  if (!outSplitSizes.empty() && outSplitSizes.size() <= kTruncatLength) {
    outSplitSizesStr = fmt::format("\"[{}]\"", fmt::join(outSplitSizes, ", "));
    metadataMap.emplace(kOutSplit, outSplitSizesStr);
  } else if (outSplitSizes.size() > kTruncatLength) {
    outSplitSizesStr = fmt::format(
        "\"[{}, ...]\"",
        fmt::join(
            outSplitSizes.begin(),
            outSplitSizes.begin() + kTruncatLength,
            ", "));
    metadataMap.emplace(kOutSplit, outSplitSizesStr);
  }

  std::vector<int64_t> groupRanks(64, 0);
  std::string groupRanksStr = "";
  if (!groupRanks.empty() && groupRanks.size() <= kTruncatLength) {
    metadataMap.emplace(
        kGroupRanks, fmt::format("\"[{}]\"", fmt::join(groupRanks, ", ")));
  } else if (groupRanks.size() > kTruncatLength) {
    metadataMap.emplace(
        kGroupRanks,
        fmt::format(
            "\"[{}, ..., {}]\"",
            fmt::join(
                groupRanks.begin(),
                groupRanks.begin() + kTruncatLength - 1,
                ", "),
            groupRanks.back()));
  }

  // Set up CPU events
  auto cpuOps = std::make_unique<MockCpuActivityBuffer>(
      start_time_ns, start_time_ns + duration_ns);
  cpuOps->addOp(
      kParamCommsCallName,
      kernelLaunchTime,
      kernelLaunchTime + 10,
      1,
      metadataMap);
  profiler.transferCpuTrace(std::move(cpuOps));

  // Set up corresponding GPU events and connect with CPU events
  // via correlationId
  auto gpuOps = std::make_unique<MockRocLogger>();
  gpuOps->addCorrelationActivity(1, RocLogger::CorrelationDomain::Domain0, 1);
  gpuOps->addKernelActivity(kernelLaunchTime + 5, kernelLaunchTime + 10, 1);
  rocActivities_.activityLogger = std::move(gpuOps);

  // Process trace
  auto logger = std::make_unique<MemoryTraceLogger>(*cfg_);
  profiler.processTrace(*logger);
  profiler.setLogger(logger.get());

  // Profiler can be reset at this point - logger owns the activities
  profiler.reset();

  // Check the content of GPU event and we should see extra
  // collective fields get populated from CPU event.
  ActivityTrace trace(std::move(logger), loggerFactory);
  EXPECT_EQ(2, trace.activities()->size());
  auto& cpu_op = trace.activities()->at(0);
  auto& gpu_kernel = trace.activities()->at(1);
  EXPECT_EQ(cpu_op->name(), kParamCommsCallName);
  EXPECT_EQ(gpu_kernel->name(), "kernel");

  // Check vector with length > 30 get truncated successfully
  std::vector<int64_t> expectedInSplit(kTruncatLength, 0);
  auto expectedInSplitStr =
      fmt::format("\"[{}, ...]\"", fmt::join(expectedInSplit, ", "));
  EXPECT_EQ(cpu_op->getMetadataValue(kInSplit), expectedInSplitStr);
  std::vector<int64_t> expectedGroupRanks(kTruncatLength - 1, 0);
  auto expectedGroupRanksStr = fmt::format(
      "\"[{}, ..., {}]\"", fmt::join(expectedGroupRanks, ", "), "0");
  EXPECT_EQ(cpu_op->getMetadataValue(kGroupRanks), expectedGroupRanksStr);

#ifdef __linux__
  // Test saved output can be loaded as JSON
  auto tmpTrace = createTempTraceFile("libkineto_test", ".json");
  LOG(INFO) << "Logging to tmp file: " << tmpTrace.path();
  trace.save(tmpTrace.path());

  // Check that the saved JSON file can be loaded and deserialized
  std::ifstream file(tmpTrace.path());
  if (!file.is_open()) {
    KINETO_THROW(std::runtime_error, "Failed to open the trace JSON file.");
  }
  std::string jsonStr(
      (std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  nlohmann::json jsonData = nlohmann::json::parse(jsonStr);

  // Convert the JSON object to a string and check
  // if the substring exists
  std::string jsonString = jsonData.dump();
  // Check that the metadata fields are present in the JSON file
  EXPECT_EQ(2, countSubstrings(jsonString, "65664"));
  EXPECT_EQ(2, countSubstrings(jsonString, kInMsgNelems));
  EXPECT_EQ(2, countSubstrings(jsonString, "65664"));
  EXPECT_EQ(2, countSubstrings(jsonString, kOutMsgNelems));
  EXPECT_EQ(2, countSubstrings(jsonString, "131328"));
  EXPECT_EQ(2, countSubstrings(jsonString, kInSplit));
  EXPECT_EQ(2, countSubstrings(jsonString, expectedInSplitStr));
  EXPECT_EQ(2, countSubstrings(jsonString, kOutSplit));
  EXPECT_EQ(2, countSubstrings(jsonString, outSplitSizesStr));
  EXPECT_EQ(2, countSubstrings(jsonString, kCollectiveName));
  EXPECT_EQ(2, countSubstrings(jsonString, "_allgather_base"));
  EXPECT_EQ(2, countSubstrings(jsonString, kProcessGroupName));
  EXPECT_EQ(2, countSubstrings(jsonString, "12341234"));
  EXPECT_EQ(2, countSubstrings(jsonString, kProcessGroupDesc));
  EXPECT_EQ(2, countSubstrings(jsonString, "test_purpose"));
  EXPECT_EQ(2, countSubstrings(jsonString, kGroupRanks));
  EXPECT_EQ(2, countSubstrings(jsonString, expectedGroupRanksStr));
#endif
}

TEST_F(RocmActivityProfilerTest, GpuUserAnnotationTest) {
  // Verbose logging is useful for debugging
  std::vector<std::string> log_modules({"RocmActivityProfiler.cpp"});
  SET_LOG_VERBOSITY_LEVEL(2, log_modules);

  // Start and stop profiling
  RocmActivityProfiler profiler(rocActivities_, /*cpu only*/ false);
  int64_t start_time_ns =
      libkineto::timeSinceEpoch(std::chrono::system_clock::now());
  int64_t duration_ns = 300;
  auto start_time = timePointFromNs(start_time_ns);
  profiler.configure(*cfg_, start_time);
  profiler.startTrace(start_time);
  profiler.stopTrace(timePointFromNs(start_time_ns + duration_ns));

  int64_t kernelLaunchTime = start_time_ns + 20;
  profiler.recordThreadInfo();

  // set up CPU event
  auto cpuOps = std::make_unique<MockCpuActivityBuffer>(
      start_time_ns, start_time_ns + duration_ns);
  cpuOps->addOp("annotation", kernelLaunchTime, kernelLaunchTime + 10, 1);
  profiler.transferCpuTrace(std::move(cpuOps));

  // set up a couple of GPU events and correlate with above CPU event.
  // RocLogger::CorrelationDomain::Domain1 is used for user annotations.
  auto gpuOps = std::make_unique<MockRocLogger>();
  gpuOps->addCorrelationActivity(1, RocLogger::CorrelationDomain::Domain1, 1);
  gpuOps->addKernelActivity(kernelLaunchTime + 5, kernelLaunchTime + 10, 1);
  gpuOps->addCorrelationActivity(1, RocLogger::CorrelationDomain::Domain1, 1);
  gpuOps->addKernelActivity(kernelLaunchTime + 15, kernelLaunchTime + 25, 1);
  rocActivities_.activityLogger = std::move(gpuOps);

  // process trace
  auto logger = std::make_unique<MemoryTraceLogger>(*cfg_);
  profiler.processTrace(*logger);

  // Profiler can be reset at this point - logger owns the activities
  profiler.reset();

  ActivityTrace trace(std::move(logger), loggerFactory);
  std::map<std::string, int> counts;
  for (auto& activity : *trace.activities()) {
    counts[activity->name()]++;
  }

  // We should now have an additional annotation activity created
  // on the GPU timeline.
  EXPECT_EQ(counts["annotation"], 2);
  EXPECT_EQ(counts["kernel"], 2);

  auto& annotation = trace.activities()->at(0);
  auto& kernel1 = trace.activities()->at(1);
  auto& kernel2 = trace.activities()->at(2);
  auto& gpu_annotation = trace.activities()->at(3);
  // Check that gpu_annotation covers both kernels
  EXPECT_EQ(gpu_annotation->type(), ActivityType::GPU_USER_ANNOTATION);
  EXPECT_EQ(gpu_annotation->timestamp(), kernel1->timestamp());
  EXPECT_EQ(
      gpu_annotation->duration(),
      kernel2->timestamp() + kernel2->duration() - kernel1->timestamp());
  EXPECT_EQ(gpu_annotation->deviceId(), kernel1->deviceId());
  EXPECT_EQ(gpu_annotation->resourceId(), kernel1->resourceId());
  EXPECT_EQ(gpu_annotation->correlationId(), annotation->correlationId());
  EXPECT_EQ(gpu_annotation->name(), annotation->name());
}

TEST_F(RocmActivityProfilerTest, SubActivityProfilers) {
  // Verbose logging is useful for debugging
  std::vector<std::string> log_modules({"RocmActivityProfiler.cpp"});
  SET_LOG_VERBOSITY_LEVEL(2, log_modules);

  // Setup example events to test
  GenericTraceActivity ev{defaultTraceSpan(), ActivityType::GLOW_RUNTIME, ""};
  ev.device = 1;
  ev.resource = 0;

  int64_t start_time_ns =
      libkineto::timeSinceEpoch(std::chrono::system_clock::now());
  int64_t duration_ns = 1000;
  auto start_time = timePointFromNs(start_time_ns);

  std::deque<GenericTraceActivity> test_activities{3, ev};
  test_activities[0].startTime = start_time_ns;
  test_activities[0].endTime = start_time_ns + 5000;
  test_activities[0].activityName = "SubGraph A execution";
  test_activities[1].startTime = start_time_ns;
  test_activities[1].endTime = start_time_ns + 2000;
  test_activities[1].activityName = "Operator foo";
  test_activities[2].startTime = start_time_ns + 2500;
  test_activities[2].endTime = start_time_ns + 2900;
  test_activities[2].activityName = "Operator bar";

  auto mock_activity_profiler =
      std::make_unique<MockActivityProfiler>(test_activities);

  // Add a child profiler and check that it works
  MockRocActivities activities;
  RocmActivityProfiler profiler(activities, /*cpu only*/ true);
  profiler.addChildActivityProfiler(std::move(mock_activity_profiler));

  profiler.configure(*cfg_, start_time);
  profiler.startTrace(start_time);
  profiler.stopTrace(timePointFromNs(start_time_ns + duration_ns));

  auto tmpTrace = createTempTraceFile("libkineto_test", ".json");
  LOG(INFO) << "Logging to tmp file " << tmpTrace.path();

  // process trace
  auto logger = std::make_unique<MemoryTraceLogger>(*cfg_);
  profiler.processTrace(*logger);
  profiler.setLogger(logger.get());

  // Profiler can be reset at this point - logger owns the activities
  profiler.reset();

  ActivityTrace trace(std::move(logger), loggerFactory);
  trace.save(tmpTrace.path());
  const auto& traced_activites = trace.activities();

  // Test we have all the events
  EXPECT_EQ(traced_activites->size(), test_activities.size());

  checkTracefile(tmpTrace.c_str());
}

TEST_F(RocmActivityProfilerTest, JsonGPUIDSortTest) {
  // Set logging level for debugging purpose
  std::vector<std::string> log_modules(
      {"RocmActivityProfiler.cpp", "output_json.cpp"});
  SET_LOG_VERBOSITY_LEVEL(2, log_modules);

  // Start and stop profiling
  RocmActivityProfiler profiler(rocActivities_, /*cpu only*/ false);
  int64_t start_time_ns =
      libkineto::timeSinceEpoch(std::chrono::system_clock::now());
  int64_t duration_ns = 500;
  auto start_time = timePointFromNs(start_time_ns);
  profiler.configure(*cfg_, start_time);
  profiler.startTrace(start_time);
  profiler.stopTrace(timePointFromNs(start_time_ns + duration_ns));
  profiler.recordThreadInfo();

  // Set up CPU events
  auto cpuOps = std::make_unique<MockCpuActivityBuffer>(
      start_time_ns, start_time_ns + duration_ns);
  cpuOps->addOp("op1", start_time_ns + 10, start_time_ns + 30, 1);
  profiler.transferCpuTrace(std::move(cpuOps));

  // Set up GPU events
  auto gpuOps = std::make_unique<MockRocLogger>();
  gpuOps->addRuntimeKernelActivity(
      HIP_LAUNCH_KERNEL, start_time_ns + 23, start_time_ns + 28, 1);
  gpuOps->addKernelActivity(start_time_ns + 50, start_time_ns + 70, 1);
  rocActivities_.activityLogger = std::move(gpuOps);

  // Process trace
  auto logger = std::make_unique<MemoryTraceLogger>(*cfg_);
  profiler.processTrace(*logger);
  profiler.setLogger(logger.get());

  // Profiler can be reset at this point - logger owns the activities
  profiler.reset();

  // Check the contents of trace matches
  ActivityTrace trace(std::move(logger), loggerFactory);
  EXPECT_EQ(3, trace.activities()->size());

#ifdef __linux__
  // Test saved output can be loaded as JSON
  auto tmpTrace = createTempTraceFile("libkineto_test", ".json");
  LOG(INFO) << "Logging to tmp file: " << tmpTrace.path();
  trace.save(tmpTrace.path());

  // Check that the saved JSON file can be loaded and deserialized
  std::ifstream file(tmpTrace.path());
  if (!file.is_open()) {
    KINETO_THROW(std::runtime_error, "Failed to open the trace JSON file.");
  }
  std::string jsonStr(
      (std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  nlohmann::json jsonData = nlohmann::json::parse(jsonStr);

  std::unordered_map<int64_t, std::string> sortLabel;
  std::unordered_map<int64_t, int64_t> sortIdx;
  for (auto& event : jsonData["traceEvents"]) {
    if (event["name"] == "process_labels" && event["tid"] == 0 &&
        event["pid"].is_number_integer()) {
      sortLabel[event["pid"].get<int64_t>()] =
          event["args"]["labels"].get<std::string>();
      LOG(INFO) << sortLabel[event["pid"].get<int64_t>()];
    }
    if (event["name"] == "process_sort_index" && event["tid"] == 0 &&
        event["pid"].is_number_integer()) {
      sortIdx[event["pid"].get<int64_t>()] =
          event["args"]["sort_index"].get<int64_t>();
      LOG(INFO) << sortIdx[event["pid"].get<int64_t>()];
    }
  }

  // Expect at least 16 GPU nodes, and 1 or more CPU nodes.
  EXPECT_LE(16, sortLabel.size());
  for (int i = 0; i < 16; i++) {
    // Check there are 16 GPU sorts (0-15) with expected sort_index.
    EXPECT_EQ("GPU " + std::to_string(i), sortLabel[i]);
    // sortIndex is gpu + kExceedMaxPid to put GPU tracks at the bottom
    // of the trace timelines.
    EXPECT_EQ(i + kExceedMaxPid, sortIdx[i]);
  }
#endif
}

namespace {
constexpr uint32_t kDispatch = ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH;

std::unique_ptr<rocprofAsyncRow> makeDispatch(uint64_t correlation) {
  return std::make_unique<rocprofAsyncRow>(
      correlation, kDispatch, 0, 0, 0, 1, 10, 20, std::string());
}

std::unique_ptr<rocprofAsyncRow> makeSdmaCopy(uint64_t correlation) {
  return std::make_unique<rocprofAsyncRow>(
      correlation,
      ROCPROFILER_BUFFER_TRACING_MEMORY_COPY,
      0,
      ROCPROFILER_MEMORY_COPY_HOST_TO_DEVICE,
      0,
      2,
      10,
      20,
      std::string());
}

rocprofCopyRow makeCopyCall(uint64_t correlation, uint32_t cid, size_t size) {
  return rocprofCopyRow(
      correlation,
      RUNTIME_DOMAIN,
      cid,
      0,
      0,
      1,
      2,
      nullptr,
      nullptr,
      size,
      hipMemcpyHostToDevice,
      reinterpret_cast<hipStream_t>(0x1000));
}

rocprofMemsetRow makeMemsetCall(uint64_t correlation, size_t size) {
  return rocprofMemsetRow(
      correlation,
      RUNTIME_DOMAIN,
      ROCPROFILER_HIP_RUNTIME_API_ID_hipMemsetAsync,
      0,
      0,
      1,
      2,
      size,
      reinterpret_cast<hipStream_t>(0x1000));
}

nlohmann::json metadataOf(const rocprofAsyncRow& row) {
  const GpuActivity activity(&row, nullptr);
  return nlohmann::json::parse("{" + activity.metadataJson() + "}");
}
} // namespace

TEST(RocmRuntimeInternalDispatchTest, CopyOperationFromApi) {
  using detail::copyOperationFromApi;
  EXPECT_EQ(
      copyOperationFromApi(
          hipMemcpyHostToDevice, ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpy),
      ROCPROFILER_MEMORY_COPY_HOST_TO_DEVICE);
  EXPECT_EQ(
      copyOperationFromApi(
          hipMemcpyDeviceToHost, ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpy),
      ROCPROFILER_MEMORY_COPY_DEVICE_TO_HOST);
  EXPECT_EQ(
      copyOperationFromApi(
          hipMemcpyDeviceToDeviceNoCU,
          ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyAsync),
      ROCPROFILER_MEMORY_COPY_DEVICE_TO_DEVICE);
  // An explicit kind wins over the API id.
  EXPECT_EQ(
      copyOperationFromApi(
          hipMemcpyHostToHost, ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyDtoD),
      ROCPROFILER_MEMORY_COPY_HOST_TO_HOST);
  // APIs without a kind argument report hipMemcpyDefault.
  EXPECT_EQ(
      copyOperationFromApi(
          hipMemcpyDefault, ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyPeerAsync),
      ROCPROFILER_MEMORY_COPY_DEVICE_TO_DEVICE);
  EXPECT_EQ(
      copyOperationFromApi(
          hipMemcpyDefault, ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyHtoDAsync),
      ROCPROFILER_MEMORY_COPY_HOST_TO_DEVICE);
  EXPECT_EQ(
      copyOperationFromApi(
          hipMemcpyDefault, ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyDtoH),
      ROCPROFILER_MEMORY_COPY_DEVICE_TO_HOST);
  EXPECT_EQ(
      copyOperationFromApi(
          hipMemcpyDefault, ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyAsync),
      ROCPROFILER_MEMORY_COPY_NONE);
}

TEST(RocmRuntimeInternalDispatchTest, MemsetTotalBytes) {
  using detail::MemsetArgs;
  using detail::memsetTotalBytes;
  MemsetArgs bytes;
  bytes.sizeBytes = 20;
  EXPECT_EQ(
      memsetTotalBytes(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemsetAsync, bytes),
      20u);
  MemsetArgs plane;
  plane.width = 100;
  plane.height = 6;
  EXPECT_EQ(
      memsetTotalBytes(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemset2D, plane),
      600u);
  MemsetArgs volume = plane;
  volume.depth = 4;
  EXPECT_EQ(
      memsetTotalBytes(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemset3D, volume),
      2400u);
  MemsetArgs count;
  count.count = 10;
  EXPECT_EQ(
      memsetTotalBytes(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemsetD8, count), 10u);
  EXPECT_EQ(
      memsetTotalBytes(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemsetD16Async, count),
      20u);
  EXPECT_EQ(
      memsetTotalBytes(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemsetD32, count),
      40u);
  EXPECT_EQ(
      memsetTotalBytes(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemset, MemsetArgs{}),
      0u);
}

TEST(RocmRuntimeInternalDispatchTest, CopyTotalBytes) {
  using detail::CopySizeArgs;
  using detail::copyTotalBytes;
  CopySizeArgs bytes;
  bytes.sizeBytes = 20;
  EXPECT_EQ(copyTotalBytes(bytes), 20u);
  CopySizeArgs array;
  array.count = 30;
  EXPECT_EQ(copyTotalBytes(array), 30u);
  CopySizeArgs plane;
  plane.width = 100;
  plane.height = 6;
  EXPECT_EQ(copyTotalBytes(plane), 600u);
  EXPECT_EQ(copyTotalBytes(CopySizeArgs{}), 0u);
}

TEST(RocmRuntimeInternalDispatchTest, ClassifiesFillAndCopyDispatches) {
  const auto stream = reinterpret_cast<hipStream_t>(0x1000);
  rocprofCopyRow copy(
      1,
      RUNTIME_DOMAIN,
      ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyDtoDAsync,
      0,
      0,
      1,
      2,
      nullptr,
      nullptr,
      4096,
      hipMemcpyDefault,
      stream);
  rocprofMemsetRow fill(
      2,
      RUNTIME_DOMAIN,
      ROCPROFILER_HIP_RUNTIME_API_ID_hipMemsetAsync,
      0,
      0,
      1,
      2,
      20,
      stream);
  rocprofMemsetRow wholeFill(
      4,
      RUNTIME_DOMAIN,
      ROCPROFILER_HIP_RUNTIME_API_ID_hipMemsetAsync,
      0,
      0,
      1,
      2,
      1024,
      stream);
  // The runtime splits the first two calls into two dispatches each and
  // issues one dispatch for the last; correlation 3 is a user kernel.
  auto copyHead = makeDispatch(1);
  auto copyTail = makeDispatch(1);
  auto fillHead = makeDispatch(2);
  auto fillTail = makeDispatch(2);
  auto kernel = makeDispatch(3);
  auto wholeFillDispatch = makeDispatch(4);
  const auto userOp = kernel->op;
  std::vector<rocprofBase*> rows{
      &copy,
      &fill,
      &wholeFill,
      copyHead.get(),
      copyTail.get(),
      fillHead.get(),
      fillTail.get(),
      kernel.get(),
      wholeFillDispatch.get()};

  detail::classifyRuntimeInternalDispatches(rows);

  for (const auto* row : {copyHead.get(), copyTail.get()}) {
    EXPECT_EQ(row->internalApi, RocLogger::RuntimeInternalMemcpy);
    EXPECT_EQ(
        row->op,
        static_cast<uint32_t>(ROCPROFILER_MEMORY_COPY_DEVICE_TO_DEVICE));
    EXPECT_EQ(detail::asyncActivityType(*row), ActivityType::GPU_MEMCPY);
  }
  for (const auto* row : {fillHead.get(), fillTail.get()}) {
    EXPECT_EQ(row->internalApi, RocLogger::RuntimeInternalMemset);
    EXPECT_EQ(detail::asyncActivityType(*row), ActivityType::GPU_MEMSET);
  }
  // Only the first dispatch of a call carries its bytes.
  EXPECT_EQ(copyHead->internalBytes, 4096u);
  EXPECT_EQ(copyTail->internalBytes, 0u);
  EXPECT_EQ(fillHead->internalBytes, 20u);
  EXPECT_EQ(fillTail->internalBytes, 0u);
  // A split call reports no bandwidth on any of its dispatches.
  for (const auto* row :
       {copyHead.get(), copyTail.get(), fillHead.get(), fillTail.get()}) {
    EXPECT_TRUE(row->internalSplit);
  }
  EXPECT_EQ(wholeFillDispatch->internalApi, RocLogger::RuntimeInternalMemset);
  EXPECT_EQ(wholeFillDispatch->internalBytes, 1024u);
  EXPECT_FALSE(wholeFillDispatch->internalSplit);
  EXPECT_EQ(kernel->internalApi, RocLogger::RuntimeInternalNone);
  EXPECT_EQ(kernel->op, userOp);
  EXPECT_EQ(
      detail::asyncActivityType(*kernel), ActivityType::CONCURRENT_KERNEL);

  // The fill dispatches take the memset call's stream.
  detail::backfillAsyncStreams(rows, [](const rocprofAsyncRow& async) {
    return detail::asyncActivityType(async) == ActivityType::GPU_MEMSET;
  });
  EXPECT_NE(fillHead->stream, 0u);
  EXPECT_EQ(fillHead->stream, fillTail->stream);
  EXPECT_EQ(kernel->stream, 0u);

  // processActivities classifies on every call; a second pass changes nothing.
  detail::classifyRuntimeInternalDispatches(rows);
  EXPECT_EQ(copyHead->internalBytes, 4096u);
  EXPECT_EQ(copyTail->internalBytes, 0u);
  EXPECT_EQ(fillHead->internalBytes, 20u);
  EXPECT_EQ(fillTail->internalBytes, 0u);
  EXPECT_TRUE(fillHead->internalSplit);
  EXPECT_EQ(wholeFillDispatch->internalBytes, 1024u);
  EXPECT_FALSE(wholeFillDispatch->internalSplit);
  EXPECT_EQ(kernel->internalApi, RocLogger::RuntimeInternalNone);
}

TEST(RocmRuntimeInternalDispatchTest, SdmaCopyRecordsTakeCallBytes) {
  auto whole =
      makeCopyCall(1, ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyAsync, 8192);
  auto split =
      makeCopyCall(2, ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyAsync, 100);
  auto wholeCopy = makeSdmaCopy(1);
  auto splitHead = makeSdmaCopy(2);
  auto splitTail = makeSdmaCopy(2);
  // Correlation 3 has no runtime copy row.
  auto orphan = makeSdmaCopy(3);
  std::vector<rocprofBase*> rows{
      &whole,
      &split,
      wholeCopy.get(),
      splitHead.get(),
      splitTail.get(),
      orphan.get()};

  for (int pass = 0; pass < 2; ++pass) {
    detail::classifyRuntimeInternalDispatches(rows);
    EXPECT_EQ(wholeCopy->internalBytes, 8192u);
    EXPECT_FALSE(wholeCopy->internalSplit);
    EXPECT_EQ(splitHead->internalBytes, 100u);
    EXPECT_EQ(splitTail->internalBytes, 0u);
    EXPECT_TRUE(splitHead->internalSplit);
    EXPECT_TRUE(splitTail->internalSplit);
    EXPECT_EQ(orphan->internalBytes, 0u);
  }
  // The record keeps its own direction and type.
  EXPECT_EQ(
      wholeCopy->op,
      static_cast<uint32_t>(ROCPROFILER_MEMORY_COPY_HOST_TO_DEVICE));
  EXPECT_EQ(detail::asyncActivityType(*wholeCopy), ActivityType::GPU_MEMCPY);

  const auto wholeMd = metadataOf(*wholeCopy);
  EXPECT_EQ(wholeMd["kind"].get<std::string>(), "HtoD");
  EXPECT_EQ(wholeMd["bytes"].get<uint64_t>(), 8192u);
  EXPECT_TRUE(wholeMd.contains("memory bandwidth (GB/s)"));
  const auto headMd = metadataOf(*splitHead);
  EXPECT_EQ(headMd["bytes"].get<uint64_t>(), 100u);
  EXPECT_FALSE(headMd.contains("memory bandwidth (GB/s)"));
  EXPECT_FALSE(metadataOf(*splitTail).contains("bytes"));
  EXPECT_FALSE(metadataOf(*orphan).contains("bytes"));
}

TEST(RocmRuntimeInternalDispatchTest, ReportsMarkedDispatches) {
  auto copy = makeCopyCall(1, ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpy, 4096);
  copy.kind = hipMemcpyDeviceToDevice;
  auto fill = makeMemsetCall(2, 20);
  auto copyDispatch = makeDispatch(1);
  auto fillHead = makeDispatch(2);
  auto fillTail = makeDispatch(2);
  auto kernel = makeDispatch(3);
  kernel->gridX = 4;
  kernel->workgroupX = 256;
  std::vector<rocprofBase*> rows{
      &copy,
      &fill,
      copyDispatch.get(),
      fillHead.get(),
      fillTail.get(),
      kernel.get()};
  detail::classifyRuntimeInternalDispatches(rows);

  const GpuActivity copyActivity(copyDispatch.get(), nullptr);
  EXPECT_EQ(copyActivity.type(), ActivityType::GPU_MEMCPY);
  EXPECT_EQ(copyActivity.name(), "Memcpy DtoD (Device -> Device)");
  const auto copyMd = metadataOf(*copyDispatch);
  EXPECT_EQ(copyMd["kind"].get<std::string>(), "DtoD");
  EXPECT_EQ(copyMd["bytes"].get<uint64_t>(), 4096u);
  EXPECT_TRUE(copyMd.contains("memory bandwidth (GB/s)"));
  EXPECT_FALSE(copyMd.contains("grid"));
  EXPECT_FALSE(copyMd.contains("block"));

  const GpuActivity fillActivity(fillHead.get(), nullptr);
  EXPECT_EQ(fillActivity.type(), ActivityType::GPU_MEMSET);
  EXPECT_EQ(fillActivity.name(), "Memset (Device)");
  const auto headMd = metadataOf(*fillHead);
  EXPECT_EQ(headMd["bytes"].get<uint64_t>(), 20u);
  // Split: no dispatch moved all the bytes, so none reports a bandwidth.
  EXPECT_FALSE(headMd.contains("memory bandwidth (GB/s)"));
  EXPECT_FALSE(headMd.contains("grid"));
  EXPECT_FALSE(headMd.contains("block"));
  const auto tailMd = metadataOf(*fillTail);
  EXPECT_FALSE(tailMd.contains("bytes"));
  EXPECT_FALSE(tailMd.contains("memory bandwidth (GB/s)"));
  EXPECT_FALSE(tailMd.contains("grid"));

  const GpuActivity kernelActivity(kernel.get(), nullptr);
  EXPECT_EQ(kernelActivity.type(), ActivityType::CONCURRENT_KERNEL);
  const auto kernelMd = metadataOf(*kernel);
  EXPECT_EQ(
      kernelMd["grid"].get<std::vector<int64_t>>(),
      (std::vector<int64_t>{4, 0, 0}));
  EXPECT_EQ(
      kernelMd["block"].get<std::vector<int64_t>>(),
      (std::vector<int64_t>{256, 0, 0}));
  EXPECT_FALSE(kernelMd.contains("bytes"));
}

TEST(RocmRuntimeInternalDispatchTest, FilterFollowsReportedType) {
  auto fill = makeMemsetCall(1, 20);
  auto fillDispatch = makeDispatch(1);
  auto kernel = makeDispatch(2);
  auto sdma = makeSdmaCopy(3);
  std::vector<rocprofBase*> rows{
      &fill, fillDispatch.get(), kernel.get(), sdma.get()};
  detail::classifyRuntimeInternalDispatches(rows);

  const auto bit = [](ActivityType type) {
    return 1u << static_cast<uint32_t>(type);
  };
  const uint32_t memsetOnly = bit(ActivityType::GPU_MEMSET);
  EXPECT_TRUE(detail::isAsyncRowLogged(*fillDispatch, memsetOnly));
  EXPECT_FALSE(detail::isAsyncRowLogged(*kernel, memsetOnly));
  EXPECT_FALSE(detail::isAsyncRowLogged(*sdma, memsetOnly));

  const uint32_t noMemset =
      bit(ActivityType::CONCURRENT_KERNEL) | bit(ActivityType::GPU_MEMCPY);
  EXPECT_FALSE(detail::isAsyncRowLogged(*fillDispatch, noMemset));
  EXPECT_TRUE(detail::isAsyncRowLogged(*kernel, noMemset));
  EXPECT_TRUE(detail::isAsyncRowLogged(*sdma, noMemset));
}

TEST(RocmRuntimeInternalDispatchTest, MemsetAndCopyCallsStartFlows) {
  const auto fill = makeMemsetCall(1, 20);
  EXPECT_TRUE(RuntimeActivity<rocprofRow>(&fill, nullptr).flowStart());
  constexpr uint32_t copyIds[] = {
      ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyDtoDAsync,
      ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyAsync_spt};
  for (const uint32_t cid : copyIds) {
    const auto copy = makeCopyCall(2, cid, 8);
    EXPECT_TRUE(RuntimeActivity<rocprofCopyRow>(&copy, nullptr).flowStart());
  }
  const rocprofRow sync(
      3,
      RUNTIME_DOMAIN,
      ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamSynchronize,
      0,
      0,
      1,
      2);
  EXPECT_FALSE(RuntimeActivity<rocprofRow>(&sync, nullptr).flowStart());
}

TEST(RocmRuntimeInternalDispatchTest, AsyncCopyRecordIsMemcpy) {
  rocprofAsyncRow copy(
      1,
      ROCPROFILER_BUFFER_TRACING_MEMORY_COPY,
      0,
      ROCPROFILER_MEMORY_COPY_HOST_TO_DEVICE,
      0,
      1,
      10,
      20,
      std::string());
  EXPECT_EQ(detail::asyncActivityType(copy), ActivityType::GPU_MEMCPY);
}
