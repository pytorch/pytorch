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
#include <stdlib.h> // NOLINT(modernize-deprecated-headers) required for malloc free

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <optional>
#include <variant>

#include "include/Config.h"
#include "include/MetadataFieldCatalog.h"
#include "include/libkineto.h"
#include "include/output_base.h"
#include "include/time_since_epoch.h"
#include "src/ActivityTrace.h"
#include "src/ApproximateClock.h"
#include "src/CuptiActivityApi.h"
#include "src/CuptiActivityProfiler.h"
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

#define CUDA_LAUNCH_KERNEL CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000
#define CUDA_MEMCPY CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020
#define CUDA_STREAM_SYNC CUPTI_RUNTIME_TRACE_CBID_cudaStreamSynchronize_v3020
#define CUDA_EVENT_SYNC CUPTI_RUNTIME_TRACE_CBID_cudaEventSynchronize_v3020

#define CU_LAUNCH_KERNEL CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel
#define CU_LAUNCH_KERNEL_EX CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx
#define CU_MEM_CREATE CUPTI_DRIVER_TRACE_CBID_cuMemCreate
#define CU_MEM_MAP CUPTI_DRIVER_TRACE_CBID_cuMemMap
#define CU_MEM_UNMAP CUPTI_DRIVER_TRACE_CBID_cuMemUnmap
#define CU_MEM_RELEASE CUPTI_DRIVER_TRACE_CBID_cuMemRelease
#define CU_MEM_EXPORT CUPTI_DRIVER_TRACE_CBID_cuMemExportToShareableHandle
#define CU_MEM_IMPORT CUPTI_DRIVER_TRACE_CBID_cuMemImportFromShareableHandle

namespace {
using RecordedMetadataValue = std::variant<
    int64_t,
    uint64_t,
    double,
    bool,
    std::string,
    std::vector<int64_t>,
    std::vector<std::string>>;

class RecordingTypedMetadataVisitor final : public ITypedMetadataVisitor {
 public:
  template <typename T>
  [[nodiscard]]
  std::optional<typename MetadataField<T>::FieldType> get(
      const MetadataField<T>& field) const {
    auto it = values_.find(std::string{field.name});
    if (it == values_.end()) {
      return std::nullopt;
    }
    return std::get<typename MetadataField<T>::FieldType>(it->second);
  }

 private:
  void visitValue(const MetadataField<int64_t>& field, int64_t value) override {
    values_[std::string{field.name}] = value;
  }

  void visitValue(const MetadataField<double>& field, double value) override {
    values_[std::string{field.name}] = value;
  }

  void visitValue(const MetadataField<bool>& field, bool value) override {
    values_[std::string{field.name}] = value;
  }

  void visitValue(
      const MetadataField<std::string>& field,
      std::string_view value) override {
    values_[std::string{field.name}] = std::string{value};
  }

  void visitValue(
      const MetadataField<std::vector<int64_t>>& field,
      const std::vector<int64_t>& value) override {
    values_[std::string{field.name}] = value;
  }

  void visitValue(
      const MetadataField<std::vector<std::string>>& field,
      const std::vector<std::string>& value) override {
    values_[std::string{field.name}] = value;
  }

  void visitValue(const MetadataField<RawJson>& field, const RawJson& value)
      override {
    values_[std::string{field.name}] = std::string{value.value};
  }

  void visitValue(const MetadataField<uint64_t>& field, uint64_t value)
      override {
    values_[std::string{field.name}] = value;
  }

  void visitValue(
      [[maybe_unused]] const MetadataField<InputShapes>& field,
      [[maybe_unused]] const InputShapes& value) override {}

  void visitUnsupported(std::string_view /*name*/) override {}

  void beginDict(std::string_view /*name*/) override {}
  void endDict() override {}

  std::map<std::string, RecordedMetadataValue> values_;
};
} // namespace

// Provides ability to easily create a few test CUPTI ops
struct MockCuptiActivityBuffer {
  void addCorrelationActivity(
      int64_t correlation,
      CUpti_ExternalCorrelationKind externalKind,
      int64_t externalId) {
    auto& act = createActivity<CUpti_ActivityExternalCorrelation>(correlation);
    act.kind = CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION;
    act.externalId = externalId;
    act.externalKind = externalKind;
    activities.push_back(reinterpret_cast<CUpti_Activity*>(&act));
  }

  void addRuntimeActivity(
      CUpti_runtime_api_trace_cbid_enum cbid,
      int64_t start_ns,
      int64_t end_ns,
      int64_t correlation) {
    auto& act =
        createActivity<CUpti_ActivityAPI>(start_ns, end_ns, correlation);
    act.kind = CUPTI_ACTIVITY_KIND_RUNTIME;
    act.cbid = cbid;
    act.threadId = threadId();
    activities.push_back(reinterpret_cast<CUpti_Activity*>(&act));
  }

  void addDriverActivity(
      CUpti_driver_api_trace_cbid_enum cbid,
      int64_t start_ns,
      int64_t end_ns,
      int64_t correlation) {
    auto& act =
        createActivity<CUpti_ActivityAPI>(start_ns, end_ns, correlation);
    act.kind = CUPTI_ACTIVITY_KIND_DRIVER;
    act.cbid = cbid;
    act.threadId = threadId();
    activities.push_back(reinterpret_cast<CUpti_Activity*>(&act));
  }

  void addKernelActivity(
      int64_t start_ns,
      int64_t end_ns,
      int64_t correlation,
      uint32_t deviceId = 0,
      uint32_t contextId = 0,
      uint32_t streamId = 1) {
    auto& act =
        createActivity<CUpti_ActivityKernelType>(start_ns, end_ns, correlation);
    act.kind = CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL;
    act.deviceId = deviceId;
    act.contextId = contextId;
    act.streamId = streamId;
    act.name = "kernel";
    act.gridX = act.gridY = act.gridZ = 1;
    act.blockX = act.blockY = act.blockZ = 1;
    activities.push_back(reinterpret_cast<CUpti_Activity*>(&act));
  }

  void addMemcpyActivity(
      int64_t start_ns,
      int64_t end_ns,
      int64_t correlation) {
    auto& act =
        createActivity<CUpti_ActivityMemcpyType>(start_ns, end_ns, correlation);
    act.kind = CUPTI_ACTIVITY_KIND_MEMCPY;
    act.deviceId = 0;
    act.streamId = 2;
    act.copyKind = CUPTI_ACTIVITY_MEMCPY_KIND_HTOD;
    act.srcKind = CUPTI_ACTIVITY_MEMORY_KIND_PINNED;
    act.dstKind = CUPTI_ACTIVITY_MEMORY_KIND_DEVICE;
    activities.push_back(reinterpret_cast<CUpti_Activity*>(&act));
  }

  void addSyncActivity(
      int64_t start_ns,
      int64_t end_ns,
      int64_t correlation,
      CUpti_ActivitySynchronizationType type,
      int64_t stream = 1,
      uint32_t cudaEventId = 0,
      uint32_t contextId = 0) {
    auto& act = createActivity<CUpti_ActivitySynchronization>(
        start_ns, end_ns, correlation);
    act.kind = CUPTI_ACTIVITY_KIND_SYNCHRONIZATION;
    act.type = type;
    act.contextId = contextId;
    act.streamId = stream;
    act.cudaEventId = cudaEventId;
    activities.push_back(reinterpret_cast<CUpti_Activity*>(&act));
  }

  void addCudaEventActivity(
      int64_t correlation,
      uint32_t eventId,
      uint32_t streamId = 1,
      uint32_t contextId = 0) {
    auto& act = createActivity<CUpti_ActivityCudaEventType>(correlation);
    act.kind = CUPTI_ACTIVITY_KIND_CUDA_EVENT;
    act.eventId = eventId;
    act.streamId = streamId;
    act.contextId = contextId;
    activities.push_back(reinterpret_cast<CUpti_Activity*>(&act));
  }

  void addCollectiveActivity(
      int64_t start_ns,
      int64_t end_ns,
      int64_t correlation) {
    auto& act =
        createActivity<CUpti_ActivityKernelType>(start_ns, end_ns, correlation);
    act.name = "collective_gpu";
    act.kind = CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL;
    act.queued = 0;
    act.deviceId = 0;
    act.contextId = 1;
    act.streamId = 0;
    act.registersPerThread = 32;
    act.staticSharedMemory = 1024;
    act.dynamicSharedMemory = 1024;
    act.gridX = act.gridY = act.gridZ = 1;
    act.blockX = act.blockY = act.blockZ = 1;
    activities.push_back(reinterpret_cast<CUpti_Activity*>(&act));
  }

  template <class T>
  T& createActivity(int64_t start_ns, int64_t end_ns, int64_t correlation) {
    T& act = *static_cast<T*>(malloc(sizeof(T)));
    std::memset(&act, 0, sizeof(act));
    act.start = start_ns;
    act.end = end_ns;
    act.correlationId = correlation;
    return act;
  }

  template <class T>
  T& createActivity(int64_t correlation) {
    T& act = *static_cast<T*>(malloc(sizeof(T)));
    std::memset(&act, 0, sizeof(act));
    act.correlationId = correlation;
    return act;
  }

  ~MockCuptiActivityBuffer() {
    for (CUpti_Activity* act : activities) {
      free(act);
    }
  }

  std::vector<CUpti_Activity*> activities;
};

// Drives the real buffer callbacks so tests can check the bookkeeping without a
// CUDA context. Two preconditions keep that from reaching into live CUPTI, and
// both are easy to break.
//
// First, the completed buffer carries a null context. At verbosity 1 and above
// bufferCompleted asks CUPTI for its dropped-record count, so a run at that
// verbosity makes a real CUPTI call with that null context. Only the paths that
// discard a buffer outright return before reaching it.
//
// Second, every test drains the allocated buffers before calling
// activityBuffers or clearActivities, both of which short-circuit when nothing
// is in flight. A test that leaves a buffer allocated will instead ask CUPTI to
// flush for real.
class CallbackCuptiActivityApi : public CuptiActivityApi {
 public:
  using CuptiActivityApi::canRejectBuffer_;

  std::pair<uint8_t*, size_t> requestBuffer(
      uint8_t* buffer = nullptr,
      size_t size = 0) {
    size_t maxNumRecords = 1;
    bufferRequested(&buffer, &size, &maxNumRecords);
    EXPECT_EQ(maxNumRecords, 0);
    return {buffer, size};
  }

  void completeBuffer(uint8_t* buffer, size_t size) {
    bufferCompleted(nullptr, 0, buffer, 0, size);
  }
};

// Mock parts of the CuptiActivityApi
class MockCuptiActivities : public CuptiActivityApi {
 public:
  bool isAvailable(uint32_t& version) const override {
    version = 0;
    return available;
  }

  const std::pair<int, size_t> processActivities(
      [[maybe_unused]] CuptiActivityBufferMap& bufferMap,
      const std::function<void(const CUpti_Activity*)>& handler) override {
    for (CUpti_Activity* act : activityBuffer->activities) {
      handler(act);
    }
    return {activityBuffer->activities.size(), 100};
  }

  std::unique_ptr<CuptiActivityBufferMap> activityBuffers() override {
    auto map = std::make_unique<CuptiActivityBufferMap>();
    auto buf = std::make_unique<CuptiActivityBuffer>(100);
    uint8_t* addr = buf->data();
    (*map)[addr] = std::move(buf);
    return map;
  }

  std::unique_ptr<MockCuptiActivityBuffer> activityBuffer;
  bool available{true};
};

TEST(CuptiActivityApiTest, KeepsReturningValidBuffersWhenRejectionUnsupported) {
  CallbackCuptiActivityApi cupti;
  cupti.canRejectBuffer_ = false;
  cupti.setMaxBufferSize(0);

  auto [firstBuffer, firstSize] = cupti.requestBuffer();
  ASSERT_NE(firstBuffer, nullptr);
  ASSERT_GT(firstSize, 0);
  EXPECT_FALSE(cupti.stopCollection);
  auto [secondBuffer, secondSize] = cupti.requestBuffer(firstBuffer, firstSize);
  EXPECT_TRUE(cupti.stopCollection);

  ASSERT_NE(secondBuffer, nullptr);
  ASSERT_GT(secondSize, 0);

  auto [thirdBuffer, thirdSize] = cupti.requestBuffer();
  ASSERT_NE(thirdBuffer, nullptr);
  EXPECT_NE(thirdBuffer, secondBuffer);

  cupti.completeBuffer(firstBuffer, firstSize);
  cupti.completeBuffer(secondBuffer, secondSize);
  cupti.completeBuffer(thirdBuffer, thirdSize);
  EXPECT_EQ(cupti.activityBuffers(), nullptr);
}

TEST(CuptiActivityApiTest, RejectsBuffersWhenSupported) {
  CallbackCuptiActivityApi cupti;
  cupti.canRejectBuffer_ = true;
  cupti.setMaxBufferSize(0);

  auto [firstBuffer, firstSize] = cupti.requestBuffer();
  ASSERT_NE(firstBuffer, nullptr);
  ASSERT_GT(firstSize, 0);
  EXPECT_FALSE(cupti.stopCollection);

  auto [secondBuffer, secondSize] = cupti.requestBuffer(firstBuffer, firstSize);
  EXPECT_TRUE(cupti.stopCollection);
  EXPECT_EQ(secondBuffer, nullptr);
  EXPECT_EQ(secondSize, 0);

  cupti.completeBuffer(firstBuffer, firstSize);
  auto [thirdBuffer, thirdSize] = cupti.requestBuffer(firstBuffer, firstSize);
  EXPECT_EQ(thirdBuffer, nullptr);
  EXPECT_EQ(thirdSize, 0);
  EXPECT_NE(cupti.activityBuffers(), nullptr);
}

TEST(CuptiActivityApiTest, ClearsCompletedBuffers) {
  CallbackCuptiActivityApi cupti;
  cupti.canRejectBuffer_ = true;
  cupti.setMaxBufferSize(0);

  auto [buffer, size] = cupti.requestBuffer();
  ASSERT_NE(buffer, nullptr);
  ASSERT_GT(size, 0);
  cupti.completeBuffer(buffer, size);
  ASSERT_FALSE(cupti.stopCollection);

  // The completion left nothing in flight, so this takes the path that has no
  // buffers to flush. It still has to drop the completed one.
  cupti.clearActivities();
  EXPECT_EQ(cupti.activityBuffers(), nullptr);
}

// Common setup / teardown and helper functions
class CuptiActivityProfilerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    profiler_ = std::make_unique<CuptiActivityProfiler>(
        cuptiActivities_, /*cpu only*/ false);
    cfg_ = std::make_unique<Config>();
    cfg_->validate(std::chrono::system_clock::now());
    loggerFactory.addProtocol("file", [](const std::string& url) {
      return std::unique_ptr<ActivityLogger>(new ChromeTraceLogger(url));
    });
  }

  std::unique_ptr<Config> cfg_;
  MockCuptiActivities cuptiActivities_;
  std::unique_ptr<CuptiActivityProfiler> profiler_;
  ActivityLoggerFactory loggerFactory;
};

TEST_F(CuptiActivityProfilerTest, UnavailableCuptiFallsBackToCpuOnly) {
  cuptiActivities_.available = false;
  const auto startTime = std::chrono::system_clock::now();
  constexpr auto duration = std::chrono::nanoseconds(300);

  profiler_->configure(*cfg_, startTime);
  profiler_->startTrace(startTime);

  const auto startTimeNs = libkineto::timeSinceEpoch(startTime);
  auto cpuOps = std::make_unique<MockCpuActivityBuffer>(
      startTimeNs, startTimeNs + duration.count());
  cpuOps->addOp("cpu_op", startTimeNs + 20, startTimeNs + 50, 1);
  profiler_->transferCpuTrace(std::move(cpuOps));
  profiler_->stopTrace(timePointFromNs(startTimeNs + duration.count()));

  auto logger = std::make_unique<MemoryTraceLogger>(*cfg_);
  profiler_->processTrace(*logger);

  ActivityTrace trace(std::move(logger), loggerFactory);
  ASSERT_EQ(trace.activities()->size(), 1);
  EXPECT_EQ(trace.activities()->front()->name(), "cpu_op");
}

TEST_F(CuptiActivityProfilerTest, SyncTrace) {
  // Verbose logging is useful for debugging
  std::vector<std::string> log_modules({"CuptiActivityProfiler.cpp"});
  SET_LOG_VERBOSITY_LEVEL(2, log_modules);

  // Start and stop profiling
  CuptiActivityProfiler profiler(cuptiActivities_, /*cpu only*/ false);
  int64_t start_time_ns =
      libkineto::timeSinceEpoch(std::chrono::system_clock::now());
  int64_t duration_ns = 300;
  auto start_time = timePointFromNs(start_time_ns);
  profiler.configure(*cfg_, start_time);
  profiler.startTrace(start_time);
  profiler.stopTrace(timePointFromNs(start_time_ns + duration_ns));
  libkineto::get_time_converter() = [](approx_time_t t) { return t; };

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

  // And some GPU ops
  auto gpuOps = std::make_unique<MockCuptiActivityBuffer>();
  gpuOps->addRuntimeActivity(
      CUDA_LAUNCH_KERNEL, start_time_ns + 33, start_time_ns + 38, 1);
  gpuOps->addRuntimeActivity(
      CUDA_MEMCPY, start_time_ns + 110, start_time_ns + 120, 2);
  gpuOps->addRuntimeActivity(
      CUDA_LAUNCH_KERNEL, start_time_ns + 130, start_time_ns + 145, 3);
  gpuOps->addDriverActivity(
      CU_LAUNCH_KERNEL, start_time_ns + 165, start_time_ns + 175, 4);
  gpuOps->addDriverActivity(
      CU_LAUNCH_KERNEL_EX, start_time_ns + 195, start_time_ns + 205, 5);
  gpuOps->addDriverActivity(
      CU_MEM_CREATE, start_time_ns + 220, start_time_ns + 230, 6);
  gpuOps->addDriverActivity(
      CU_MEM_MAP, start_time_ns + 235, start_time_ns + 245, 7);
  gpuOps->addDriverActivity(
      CU_MEM_UNMAP, start_time_ns + 250, start_time_ns + 260, 8);
  gpuOps->addDriverActivity(
      CU_MEM_RELEASE, start_time_ns + 265, start_time_ns + 275, 9);
  gpuOps->addDriverActivity(
      CU_MEM_EXPORT, start_time_ns + 278, start_time_ns + 285, 10);
  gpuOps->addDriverActivity(
      CU_MEM_IMPORT, start_time_ns + 287, start_time_ns + 293, 11);
  gpuOps->addRuntimeActivity(
      CUDA_STREAM_SYNC, start_time_ns + 146, start_time_ns + 240, 12);
  gpuOps->addRuntimeActivity(
      CUDA_EVENT_SYNC, start_time_ns + 241, start_time_ns + 250, 13);
  gpuOps->addKernelActivity(start_time_ns + 50, start_time_ns + 70, 1);
  gpuOps->addMemcpyActivity(start_time_ns + 140, start_time_ns + 150, 2);
  gpuOps->addKernelActivity(start_time_ns + 160, start_time_ns + 220, 3);
  gpuOps->addKernelActivity(start_time_ns + 230, start_time_ns + 250, 4);
  gpuOps->addKernelActivity(start_time_ns + 260, start_time_ns + 280, 5);
  gpuOps->addSyncActivity(
      start_time_ns + 221,
      start_time_ns + 223,
      12,
      CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_STREAM_SYNCHRONIZE);
  // Add wait event on kernel stream 1
  gpuOps->addSyncActivity(
      start_time_ns + 224,
      start_time_ns + 226,
      13,
      CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_STREAM_WAIT_EVENT,
      1 /*stream*/);
  // This event should be ignored because it is not on a stream that has no GPU
  // kernels
  gpuOps->addSyncActivity(
      start_time_ns + 226,
      start_time_ns + 230,
      14,
      CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_STREAM_WAIT_EVENT,
      4 /*stream*/);
  // Comes from CudaEventSynchronize call on CPU
  gpuOps->addSyncActivity(
      start_time_ns + 227,
      start_time_ns + 226,
      13,
      CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_EVENT_SYNCHRONIZE,
      -1 /*stream*/);
  cuptiActivities_.activityBuffer = std::move(gpuOps);

  // Have the profiler process them
  auto logger = std::make_unique<MemoryTraceLogger>(*cfg_);
  profiler.processTrace(*logger);

  // Profiler can be reset at this point - logger owns the activities
  profiler.reset();

  // Wrapper that allows iterating over the activities
  ActivityTrace trace(std::move(logger), loggerFactory);
  EXPECT_EQ(trace.activities()->size(), 26);
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
  EXPECT_EQ(activityCounts["op1"], 1);
  EXPECT_EQ(activityCounts["op2"], 1);
  EXPECT_EQ(activityCounts["op3"], 1);
  EXPECT_EQ(activityCounts["op4"], 1);
  EXPECT_EQ(activityCounts["cudaLaunchKernel"], 2);
  EXPECT_EQ(activityCounts["cuLaunchKernelEx"], 1);
  EXPECT_EQ(activityCounts["cuMemCreate"], 1);
  EXPECT_EQ(activityCounts["cuMemMap"], 1);
  EXPECT_EQ(activityCounts["cuMemUnmap"], 1);
  EXPECT_EQ(activityCounts["cuMemRelease"], 1);
  EXPECT_EQ(activityCounts["cuMemExportToShareableHandle"], 1);
  EXPECT_EQ(activityCounts["cuMemImportFromShareableHandle"], 1);
  EXPECT_EQ(activityCounts["cudaMemcpy"], 1);
  EXPECT_EQ(activityCounts["cudaStreamSynchronize"], 1);
  EXPECT_EQ(activityCounts["cudaEventSynchronize"], 1);
  EXPECT_EQ(activityCounts["kernel"], 4);
  EXPECT_EQ(activityCounts["Stream Sync"], 1);
  EXPECT_EQ(activityCounts["Event Sync"], 1);
  EXPECT_EQ(activityCounts["Memcpy HtoD (Pinned -> Device)"], 1);

  auto sysTid = systemThreadId();
  // Ops and runtime events are on thread sysTid along with the flow start
  // events
  EXPECT_EQ(resourceIds[sysTid], 18);
  // Kernels and sync events are on stream 1, memcpy on stream 2
  EXPECT_EQ(resourceIds[1], 6);
  EXPECT_EQ(resourceIds[2], 1);

#ifdef __linux__
  auto tmpTrace = createTempTraceFile("libkineto_test", ".json");
  trace.save(tmpTrace.path());
  checkTracefile(tmpTrace.c_str());

  // Verify Stream Sync events are on a separate row from kernel events
  // in the JSON trace output (tid offset by kSyncStreamTidOffset).
  {
    std::ifstream traceFile(tmpTrace.path());
    std::string traceStr(
        (std::istreambuf_iterator<char>(traceFile)),
        std::istreambuf_iterator<char>());
    auto traceJson = nlohmann::json::parse(traceStr);
    int64_t kernelTid = -1;
    int64_t streamSyncTid = -1;
    bool foundSyncRowMeta = false;
    for (const auto& event : traceJson["traceEvents"]) {
      if (event.value("name", "") == "Kernel") {
        kernelTid = event.value("tid", (int64_t)-1);
      }
      if (event.value("name", "") == "Stream Sync") {
        streamSyncTid = event.value("tid", (int64_t)-1);
      }
      if (event.value("name", "") == "thread_name") {
        auto args = event.value("args", nlohmann::json::object());
        std::string threadName = args.value("name", "");
        if (threadName.find("sync") != std::string::npos) {
          foundSyncRowMeta = true;
        }
      }
    }
    EXPECT_NE(kernelTid, -1) << "Expected kernel events in trace";
    EXPECT_NE(streamSyncTid, -1) << "Expected Stream Sync event in trace";
    EXPECT_NE(kernelTid, streamSyncTid)
        << "Stream Sync should be on a different tid than kernels";
    EXPECT_TRUE(foundSyncRowMeta)
        << "Expected thread_name metadata for sync row";
  }
#endif
}

TEST_F(CuptiActivityProfilerTest, SyncEventCorrIdOutOfOrder) {
  // Test that wait_on_cuda_event_record_corr_id is populated even when
  // SYNCHRONIZATION records appear before both the CUDA_EVENT and the kernel
  // that provides the context->device mapping.
  std::vector<std::string> log_modules({"CuptiActivityProfiler.cpp"});
  SET_LOG_VERBOSITY_LEVEL(2, log_modules);

  CuptiActivityProfiler profiler(cuptiActivities_, /*cpu only*/ false);
  int64_t start_time_ns =
      libkineto::timeSinceEpoch(std::chrono::system_clock::now());
  int64_t duration_ns = 300;
  auto start_time = timePointFromNs(start_time_ns);
  profiler.configure(*cfg_, start_time);
  profiler.startTrace(start_time);
  profiler.stopTrace(timePointFromNs(start_time_ns + duration_ns));
  libkineto::get_time_converter() = [](approx_time_t t) { return t; };

  profiler.recordThreadInfo();

  auto cpuOps = std::make_unique<MockCpuActivityBuffer>(
      start_time_ns, start_time_ns + duration_ns);
  cpuOps->addOp("op1", start_time_ns + 10, start_time_ns + 50, 1);
  cpuOps->addOp("op_record", start_time_ns + 60, start_time_ns + 80, 100);
  cpuOps->addOp("op_wait", start_time_ns + 90, start_time_ns + 110, 200);
  cpuOps->addOp("op_evt_sync", start_time_ns + 120, start_time_ns + 140, 300);
  profiler.transferCpuTrace(std::move(cpuOps));

  constexpr uint32_t kEventId = 7777;
  constexpr uint32_t kContextId = 7;
  constexpr uint32_t kDeviceId = 3;
  constexpr uint32_t kRecordCorrId = 100;
  constexpr uint32_t kWaitCorrId = 200;
  constexpr uint32_t kEvtSyncCorrId = 300;
  constexpr uint32_t kEventStreamId = 11;
  constexpr uint32_t kWaitStreamId = 13;

  // Wait events and synchronization records are added
  // before the CUDA_EVENT and kernel they reference, as CUPTI
  // provides no ordering guarantee for activity buffer entries.
  auto gpuOps = std::make_unique<MockCuptiActivityBuffer>();
  gpuOps->addRuntimeActivity(
      CUDA_LAUNCH_KERNEL, start_time_ns + 10, start_time_ns + 20, 1);
  gpuOps->addSyncActivity(
      start_time_ns + 100,
      start_time_ns + 110,
      kWaitCorrId,
      CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_STREAM_WAIT_EVENT,
      kWaitStreamId,
      kEventId,
      kContextId);
  gpuOps->addSyncActivity(
      start_time_ns + 120,
      start_time_ns + 140,
      kEvtSyncCorrId,
      CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_EVENT_SYNCHRONIZE,
      -1,
      kEventId,
      kContextId);
  gpuOps->addCudaEventActivity(
      kRecordCorrId, kEventId, kEventStreamId, kContextId);
  gpuOps->addKernelActivity(
      start_time_ns + 30,
      start_time_ns + 50,
      1,
      kDeviceId,
      kContextId,
      kWaitStreamId);
  cuptiActivities_.activityBuffer = std::move(gpuOps);

  auto logger = std::make_unique<MemoryTraceLogger>(*cfg_);
  profiler.processTrace(*logger);
  profiler.reset();

  ActivityTrace trace(std::move(logger), loggerFactory);

  // Find the sync activities and check their metadata
  int streamWaitFound = 0;
  int eventSyncFound = 0;
  for (auto& activity : *trace.activities()) {
    std::string metadata = activity->metadataJson();
    if (metadata.find("Stream Wait Event") != std::string::npos) {
      auto json = nlohmann::json::parse("{" + metadata + "}");
      EXPECT_EQ(json["wait_on_cuda_event_id"], kEventId)
          << "Stream Wait Event should reference the correct event ID";
      EXPECT_EQ(json["wait_on_cuda_event_record_corr_id"], kRecordCorrId)
          << "Stream Wait Event corr_id should be populated despite out-of-order records";
      EXPECT_EQ(json["wait_on_stream"], kEventStreamId)
          << "Stream Wait Event should reference stream the event was recorded on";
      RecordingTypedMetadataVisitor typedMetadata;
      activity->visitTypedMetadata(typedMetadata);
      EXPECT_EQ(
          typedMetadata.get(CudaMetadataFields::kWaitOnCudaEventId),
          static_cast<uint64_t>(kEventId));
      EXPECT_EQ(
          typedMetadata.get(CudaMetadataFields::kWaitOnCudaEventRecordCorrId),
          static_cast<int64_t>(kRecordCorrId));
      EXPECT_EQ(
          typedMetadata.get(CudaMetadataFields::kWaitOnStream),
          static_cast<int64_t>(kEventStreamId));
      streamWaitFound++;
    }
    if (metadata.find("Event Sync") != std::string::npos) {
      auto json = nlohmann::json::parse("{" + metadata + "}");
      EXPECT_EQ(json["wait_on_cuda_event_id"], kEventId)
          << "Event Sync should reference the correct event ID";
      EXPECT_EQ(json["wait_on_cuda_event_record_corr_id"], kRecordCorrId)
          << "Event Sync corr_id should be populated despite out-of-order records";
      EXPECT_EQ(json["wait_on_stream"], kEventStreamId)
          << "Event Sync should reference stream the event was recorded on";
      RecordingTypedMetadataVisitor typedMetadata;
      activity->visitTypedMetadata(typedMetadata);
      EXPECT_EQ(
          typedMetadata.get(CudaMetadataFields::kWaitOnCudaEventId),
          static_cast<uint64_t>(kEventId));
      EXPECT_EQ(
          typedMetadata.get(CudaMetadataFields::kWaitOnCudaEventRecordCorrId),
          static_cast<int64_t>(kRecordCorrId));
      EXPECT_EQ(
          typedMetadata.get(CudaMetadataFields::kWaitOnStream),
          static_cast<int64_t>(kEventStreamId));
      eventSyncFound++;
    }
  }
  EXPECT_EQ(streamWaitFound, 1) << "Expected exactly one Stream Wait Event";
  EXPECT_EQ(eventSyncFound, 1) << "Expected exactly one Event Sync";

#ifdef __linux__
  auto tmpTrace = createTempTraceFile("libkineto_out_of_order_", ".json");
  trace.save(tmpTrace.path());
  LOG(INFO) << "Trace exported to: " << tmpTrace.path();
#endif
}

TEST_F(CuptiActivityProfilerTest, GpuNCCLCollectiveTest) {
  // Set logging level for debugging purpose
  std::vector<std::string> log_modules(
      {"CuptiActivityProfiler.cpp", "output_json.cpp"});
  SET_LOG_VERBOSITY_LEVEL(2, log_modules);

  // Start and stop profiling
  CuptiActivityProfiler profiler(cuptiActivities_, /*cpu only*/ false);
  int64_t start_time_ns =
      libkineto::timeSinceEpoch(std::chrono::system_clock::now());
  int64_t duration_ns = 300;
  auto start_time = timePointFromNs(start_time_ns);
  profiler.configure(*cfg_, start_time);
  profiler.startTrace(start_time);
  profiler.stopTrace(timePointFromNs(start_time_ns + duration_ns));
  libkineto::get_time_converter() = [](approx_time_t t) { return t; };

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
  metadataMap.emplace(kSeqNum, "4242424242");
  metadataMap.emplace(kCommsId, "12345678");

  std::vector<int64_t> inSplitSizes(50, 0);
  std::string inSplitSizesStr;
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
  std::string outSplitSizesStr;
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
  std::string groupRanksStr;
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

  // Set up GPU events with two collectives: one after its correlation
  // record (in-order) and one before (out-of-order), to verify metadata
  // propagation works regardless of CUPTI buffer ordering.
  auto gpuOps = std::make_unique<MockCuptiActivityBuffer>();
  gpuOps->addCorrelationActivity(1, CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0, 1);
  gpuOps->addCollectiveActivity(kernelLaunchTime + 5, kernelLaunchTime + 10, 1);
  gpuOps->addCollectiveActivity(
      kernelLaunchTime + 15, kernelLaunchTime + 20, 2);
  gpuOps->addCorrelationActivity(2, CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0, 1);
  cuptiActivities_.activityBuffer = std::move(gpuOps);

  // Process trace
  auto logger = std::make_unique<MemoryTraceLogger>(*cfg_);
  profiler.processTrace(*logger);
  profiler.setLogger(logger.get());

  // Profiler can be reset at this point - logger owns the activities
  profiler.reset();

  // Check the content of GPU event and we should see extra
  // collective fields get populated from CPU event.
  ActivityTrace trace(std::move(logger), loggerFactory);
  EXPECT_EQ(3, trace.activities()->size());
  auto& cpu_annotation = trace.activities()->at(0);
  auto& gpu_annotation1 = trace.activities()->at(1);
  auto& gpu_annotation2 = trace.activities()->at(2);
  EXPECT_EQ(cpu_annotation->name(), kParamCommsCallName);
  EXPECT_EQ(gpu_annotation1->name(), "collective_gpu");
  EXPECT_EQ(gpu_annotation2->name(), "collective_gpu");

  // Check vector with length > 30 get truncated successfully
  std::vector<int64_t> expectedInSplit(kTruncatLength, 0);
  auto expectedInSplitStr =
      fmt::format("\"[{}, ...]\"", fmt::join(expectedInSplit, ", "));
  EXPECT_EQ(cpu_annotation->getMetadataValue(kInSplit), expectedInSplitStr);
  std::vector<int64_t> expectedGroupRanks(kTruncatLength - 1, 0);
  auto expectedGroupRanksStr = fmt::format(
      "\"[{}, ..., {}]\"", fmt::join(expectedGroupRanks, ", "), "0");
  EXPECT_EQ(
      cpu_annotation->getMetadataValue(kGroupRanks), expectedGroupRanksStr);

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
  EXPECT_EQ(3, countSubstrings(jsonString, "65664"));
  EXPECT_EQ(3, countSubstrings(jsonString, kInMsgNelems));
  EXPECT_EQ(3, countSubstrings(jsonString, "65664"));
  EXPECT_EQ(3, countSubstrings(jsonString, kOutMsgNelems));
  EXPECT_EQ(3, countSubstrings(jsonString, "131328"));
  EXPECT_EQ(3, countSubstrings(jsonString, kInSplit));
  EXPECT_EQ(3, countSubstrings(jsonString, expectedInSplitStr));
  EXPECT_EQ(3, countSubstrings(jsonString, kOutSplit));
  EXPECT_EQ(3, countSubstrings(jsonString, outSplitSizesStr));
  EXPECT_EQ(3, countSubstrings(jsonString, kCollectiveName));
  EXPECT_EQ(3, countSubstrings(jsonString, "_allgather_base"));
  EXPECT_EQ(3, countSubstrings(jsonString, kProcessGroupName));
  EXPECT_EQ(3, countSubstrings(jsonString, "12341234"));
  EXPECT_EQ(3, countSubstrings(jsonString, kProcessGroupDesc));
  EXPECT_EQ(3, countSubstrings(jsonString, "test_purpose"));
  EXPECT_EQ(3, countSubstrings(jsonString, kGroupRanks));
  EXPECT_EQ(3, countSubstrings(jsonString, expectedGroupRanksStr));
  EXPECT_EQ(3, countSubstrings(jsonString, kSeqNum));
  EXPECT_EQ(3, countSubstrings(jsonString, "4242424242"));
  EXPECT_EQ(3, countSubstrings(jsonString, kCommsId));
  EXPECT_EQ(3, countSubstrings(jsonString, "12345678"));
#endif
}

TEST_F(CuptiActivityProfilerTest, GpuUserAnnotationTest) {
  // Verbose logging is useful for debugging
  std::vector<std::string> log_modules({"CuptiActivityProfiler.cpp"});
  SET_LOG_VERBOSITY_LEVEL(2, log_modules);

  // Start and stop profiling
  CuptiActivityProfiler profiler(cuptiActivities_, /*cpu only*/ false);
  int64_t start_time_ns =
      libkineto::timeSinceEpoch(std::chrono::system_clock::now());
  int64_t duration_ns = 300;
  auto start_time = timePointFromNs(start_time_ns);
  profiler.configure(*cfg_, start_time);
  profiler.startTrace(start_time);
  profiler.stopTrace(timePointFromNs(start_time_ns + duration_ns));
  libkineto::get_time_converter() = [](approx_time_t t) { return t; };

  int64_t kernelLaunchTime = start_time_ns + 20;
  profiler.recordThreadInfo();

  // set up CPU event
  auto cpuOps = std::make_unique<MockCpuActivityBuffer>(
      start_time_ns, start_time_ns + duration_ns);
  cpuOps->addOp("annotation", kernelLaunchTime, kernelLaunchTime + 10, 1);
  profiler.transferCpuTrace(std::move(cpuOps));

  // set up a couple of GPU events and correlate with above CPU event.
  // CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM1 is used for user annotations.
  auto gpuOps = std::make_unique<MockCuptiActivityBuffer>();
  gpuOps->addCorrelationActivity(1, CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM1, 1);
  gpuOps->addKernelActivity(kernelLaunchTime + 5, kernelLaunchTime + 10, 1);
  gpuOps->addCorrelationActivity(1, CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM1, 1);
  gpuOps->addKernelActivity(kernelLaunchTime + 15, kernelLaunchTime + 25, 1);
  cuptiActivities_.activityBuffer = std::move(gpuOps);

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

TEST_F(CuptiActivityProfilerTest, SubActivityProfilers) {
  // Verbose logging is useful for debugging
  std::vector<std::string> log_modules({"CuptiActivityProfiler.cpp"});
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

  MockCuptiActivities activities;
  CuptiActivityProfiler profiler(activities, /*cpu only*/ true);
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

TEST_F(CuptiActivityProfilerTest, JsonGPUIDSortTest) {
  // Set logging level for debugging purpose
  std::vector<std::string> log_modules(
      {"CuptiActivityProfiler.cpp", "output_json.cpp"});
  SET_LOG_VERBOSITY_LEVEL(2, log_modules);

  // Start and stop profiling
  CuptiActivityProfiler profiler(cuptiActivities_, /*cpu only*/ false);
  int64_t start_time_ns =
      libkineto::timeSinceEpoch(std::chrono::system_clock::now());
  int64_t duration_ns = 500;
  auto start_time = timePointFromNs(start_time_ns);
  profiler.configure(*cfg_, start_time);
  profiler.startTrace(start_time);
  profiler.stopTrace(timePointFromNs(start_time_ns + duration_ns));
  libkineto::get_time_converter() = [](approx_time_t t) { return t; };
  profiler.recordThreadInfo();

  // Set up CPU events and corresponding GPU events
  auto cpuOps = std::make_unique<MockCpuActivityBuffer>(
      start_time_ns, start_time_ns + duration_ns);
  cpuOps->addOp("op1", start_time_ns + 10, start_time_ns + 30, 1);
  profiler.transferCpuTrace(std::move(cpuOps));
  auto gpuOps = std::make_unique<MockCuptiActivityBuffer>();
  gpuOps->addRuntimeActivity(
      CUDA_LAUNCH_KERNEL, start_time_ns + 23, start_time_ns + 28, 1);
  gpuOps->addKernelActivity(start_time_ns + 50, start_time_ns + 70, 1);
  cuptiActivities_.activityBuffer = std::move(gpuOps);

  // Process trace
  auto logger = std::make_unique<MemoryTraceLogger>(*cfg_);
  profiler.processTrace(*logger);
  profiler.setLogger(logger.get());

  // Profiler can be reset at this point - logger owns the activities
  profiler.reset();

  // Check the content of GPU event and we should see extra
  // collective fields get populated from CPU event.
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
    }
    if (event["name"] == "process_sort_index" && event["tid"] == 0 &&
        event["pid"].is_number_integer()) {
      sortIdx[event["pid"].get<int64_t>()] =
          event["args"]["sort_index"].get<int64_t>();
    }
  }

  // Expect there is 1 CUPTI Overhead, and 16 CPU + GPU sorts, total 17.
  EXPECT_EQ(17, sortLabel.size());
  for (int i = 0; i < 16; i++) {
    // Check there are 16 GPU sorts (0-15) with expected sort_index.
    EXPECT_EQ("GPU " + std::to_string(i), sortLabel[i]);
    // sortIndex is gpu + kExceedMaxPid to put GPU tracks at the bottom
    // of the trace timelines.
    EXPECT_EQ(i + kExceedMaxPid, sortIdx[i]);
  }
#endif
}

TEST_F(CuptiActivityProfilerTest, StreamWaitEventFutureCorrelation) {
  // When a CUDA event is re-recorded (same eventId, new correlationId), CUPTI
  // may deliver the later record before the wait. Verify the wait links to the
  // most recent record that precedes it, not the future re-record.
  std::vector<std::string> log_modules({"CuptiActivityProfiler.cpp"});
  SET_LOG_VERBOSITY_LEVEL(2, log_modules);

  CuptiActivityProfiler profiler(cuptiActivities_, /*cpu only*/ false);
  int64_t start_time_ns =
      libkineto::timeSinceEpoch(std::chrono::system_clock::now());
  int64_t duration_ns = 500;
  auto start_time = timePointFromNs(start_time_ns);
  profiler.configure(*cfg_, start_time);
  profiler.startTrace(start_time);
  profiler.stopTrace(timePointFromNs(start_time_ns + duration_ns));
  libkineto::get_time_converter() = [](approx_time_t t) { return t; };
  profiler.recordThreadInfo();

  auto cpuOps = std::make_unique<MockCpuActivityBuffer>(
      start_time_ns, start_time_ns + duration_ns);
  cpuOps->addOp("op1", start_time_ns + 10, start_time_ns + 30, 1);
  profiler.transferCpuTrace(std::move(cpuOps));

  // Chronological order: record(corrId=100), wait(corrId=101),
  // re-record(corrId=200). Delivery order: record(100), re-record(200),
  // wait(101).
  auto gpuOps = std::make_unique<MockCuptiActivityBuffer>();
  gpuOps->addRuntimeActivity(
      CUDA_LAUNCH_KERNEL, start_time_ns + 13, start_time_ns + 18, 1);
  gpuOps->addKernelActivity(start_time_ns + 50, start_time_ns + 70, 1);
  gpuOps->addCudaEventActivity(100, 42, 1, 0);
  gpuOps->addCudaEventActivity(200, 42, 1, 0);
  gpuOps->addSyncActivity(
      start_time_ns + 200,
      start_time_ns + 202,
      101,
      CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_STREAM_WAIT_EVENT,
      1,
      42);
  cuptiActivities_.activityBuffer = std::move(gpuOps);

  auto logger = std::make_unique<MemoryTraceLogger>(*cfg_);
  profiler.processTrace(*logger);
  profiler.reset();

  ActivityTrace trace(std::move(logger), loggerFactory);

  bool foundWaitEvent = false;
  for (auto& activity : *trace.activities()) {
    if (activity->name() == "Stream Wait Event") {
      foundWaitEvent = true;
      auto metadata = activity->metadataJson();
      auto json = nlohmann::json::parse("{" + metadata + "}");
      EXPECT_EQ(json["wait_on_cuda_event_record_corr_id"], 100)
          << "Should reference corrId 100 (the record before the wait), got: "
          << metadata;
      EXPECT_EQ(json["wait_on_cuda_event_id"], 42)
          << "Should reference eventId 42, got: " << metadata;
    }
  }
  EXPECT_TRUE(foundWaitEvent) << "Stream Wait Event activity not found";
}

TEST_F(CuptiActivityProfilerTest, WaitEventMapClearedOnReset) {
  // Verify waitEventMap is cleared between profiling sessions so that a wait
  // event in session 2 does not pick up a stale record from session 1.
  std::vector<std::string> log_modules({"CuptiActivityProfiler.cpp"});
  SET_LOG_VERBOSITY_LEVEL(2, log_modules);

  int64_t start_time_ns =
      libkineto::timeSinceEpoch(std::chrono::system_clock::now());
  int64_t duration_ns = 500;
  auto start_time = timePointFromNs(start_time_ns);

  // Session 1: record eventId=42 with corrId=100, then reset.
  {
    CuptiActivityProfiler profiler(cuptiActivities_, /*cpu only*/ false);
    profiler.configure(*cfg_, start_time);
    profiler.startTrace(start_time);
    profiler.stopTrace(timePointFromNs(start_time_ns + duration_ns));
    libkineto::get_time_converter() = [](approx_time_t t) { return t; };
    profiler.recordThreadInfo();

    auto cpuOps = std::make_unique<MockCpuActivityBuffer>(
        start_time_ns, start_time_ns + duration_ns);
    cpuOps->addOp("op1", start_time_ns + 10, start_time_ns + 30, 1);
    profiler.transferCpuTrace(std::move(cpuOps));

    auto gpuOps = std::make_unique<MockCuptiActivityBuffer>();
    gpuOps->addRuntimeActivity(
        CUDA_LAUNCH_KERNEL, start_time_ns + 13, start_time_ns + 18, 1);
    gpuOps->addKernelActivity(start_time_ns + 50, start_time_ns + 70, 1);
    gpuOps->addCudaEventActivity(100, 42, 1, 0);
    cuptiActivities_.activityBuffer = std::move(gpuOps);

    auto logger = std::make_unique<MemoryTraceLogger>(*cfg_);
    profiler.processTrace(*logger);
    profiler.reset();
  }

  // Session 2: no cudaEventRecord for eventId=42, but a wait references it.
  {
    CuptiActivityProfiler profiler2(cuptiActivities_, /*cpu only*/ false);
    int64_t start_time_ns2 = start_time_ns + 10000;
    auto start_time2 = timePointFromNs(start_time_ns2);

    auto cfg2 = std::make_unique<Config>();
    cfg2->validate(std::chrono::system_clock::now());

    profiler2.configure(*cfg2, start_time2);
    profiler2.startTrace(start_time2);
    profiler2.stopTrace(timePointFromNs(start_time_ns2 + duration_ns));
    libkineto::get_time_converter() = [](approx_time_t t) { return t; };
    profiler2.recordThreadInfo();

    auto cpuOps2 = std::make_unique<MockCpuActivityBuffer>(
        start_time_ns2, start_time_ns2 + duration_ns);
    cpuOps2->addOp("op1", start_time_ns2 + 10, start_time_ns2 + 30, 1);
    profiler2.transferCpuTrace(std::move(cpuOps2));

    auto gpuOps2 = std::make_unique<MockCuptiActivityBuffer>();
    gpuOps2->addRuntimeActivity(
        CUDA_LAUNCH_KERNEL, start_time_ns2 + 13, start_time_ns2 + 18, 1);
    gpuOps2->addKernelActivity(start_time_ns2 + 50, start_time_ns2 + 70, 1);
    gpuOps2->addSyncActivity(
        start_time_ns2 + 200,
        start_time_ns2 + 202,
        501,
        CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_STREAM_WAIT_EVENT,
        1,
        42);
    cuptiActivities_.activityBuffer = std::move(gpuOps2);

    auto logger2 = std::make_unique<MemoryTraceLogger>(*cfg2);
    profiler2.processTrace(*logger2);
    profiler2.reset();

    ActivityTrace trace2(std::move(logger2), loggerFactory);

    for (auto& activity : *trace2.activities()) {
      if (activity->name() == "Stream Wait Event") {
        auto metadata = activity->metadataJson();
        auto json = nlohmann::json::parse("{" + metadata + "}");
        EXPECT_EQ(json["wait_on_cuda_event_record_corr_id"], -1)
            << "Expected default corrId -1 (no record in session 2), got: "
            << metadata;
      }
    }
  }
}
