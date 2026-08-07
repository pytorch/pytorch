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
  virtual int processActivities(
      std::function<void(const rocprofBase*)> handler,
      std::function<void(uint64_t, uint64_t, RocLogger::CorrelationDomain)>
          correlationHandler) override {
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
    detail::backfillAsyncStreams(
        activityLogger->activities_, [](const rocprofAsyncRow& async) {
          return isAsyncCopy(async) || isAsyncKernel(async);
        });
    for (auto& item : activityLogger->activities_) {
      handler(item);
      ++count;
    }
    return count;
  }

  std::unique_ptr<MockRocLogger> activityLogger;
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

TEST_F(RocmActivityProfilerTest, SyncTrace) {
  // Verbose logging is useful for debugging
  std::vector<std::string> log_modules({"RocmActivityProfiler.cpp"});
  SET_LOG_VERBOSITY_LEVEL(2, log_modules);

  // Start and stop profiling
  RocmActivityProfiler profiler(rocActivities_, /*cpu only*/ false);
  int64_t start_time_ns =
      libkineto::timeSinceEpoch(std::chrono::system_clock::now());
  int64_t duration_ns = 300;
  auto start_time = time_point<system_clock>(nanoseconds(start_time_ns));
  profiler.configure(*cfg_, start_time);
  profiler.startTrace(start_time);
  profiler.stopTrace(start_time + nanoseconds(duration_ns));

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
  auto start_time = time_point<system_clock>(nanoseconds(start_time_ns));
  profiler.configure(*cfg_, start_time);
  profiler.startTrace(start_time);
  profiler.stopTrace(start_time + nanoseconds(duration_ns));

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
  auto start_time = time_point<system_clock>(nanoseconds(start_time_ns));
  profiler.configure(*cfg_, start_time);
  profiler.startTrace(start_time);
  profiler.stopTrace(start_time + nanoseconds(duration_ns));

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
  auto start_time = time_point<system_clock>(nanoseconds(start_time_ns));
  profiler.configure(*cfg_, start_time);
  profiler.startTrace(start_time);
  profiler.stopTrace(start_time + nanoseconds(duration_ns));

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
  auto start_time = time_point<system_clock>(nanoseconds(start_time_ns));
  profiler.configure(*cfg_, start_time);
  profiler.startTrace(start_time);
  profiler.stopTrace(start_time + nanoseconds(duration_ns));

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
  auto start_time = time_point<system_clock>(nanoseconds(start_time_ns));
  profiler.configure(*cfg_, start_time);
  profiler.startTrace(start_time);
  profiler.stopTrace(start_time + nanoseconds(duration_ns));

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
  auto start_time = time_point<system_clock>(nanoseconds(start_time_ns));
  profiler.configure(*cfg_, start_time);
  profiler.startTrace(start_time);
  profiler.stopTrace(start_time + nanoseconds(duration_ns));

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
  auto start_time = time_point<system_clock>(nanoseconds(start_time_ns));

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
  profiler.stopTrace(start_time + nanoseconds(duration_ns));

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
  auto start_time = time_point<system_clock>(nanoseconds(start_time_ns));
  profiler.configure(*cfg_, start_time);
  profiler.startTrace(start_time);
  profiler.stopTrace(start_time + nanoseconds(duration_ns));
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

  // Expect atleast 16 GPU nodes, and 1 or more CPU nodes.
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
