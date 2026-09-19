/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>
#include <cupti_pmsampling.h>
#include <cupti_profiler_host.h>
#include <cupti_profiler_target.h>
#include <cupti_target.h>

#include "src/CuptiPMSamplingApi.h"

using namespace KINETO_NAMESPACE;
using namespace std::chrono_literals;

namespace {

// CuptiPMSamplingApi calls CUDA and CUPTI directly. These overrides provide the
// outputs needed by the tests without touching a GPU.
struct FakeSample {
  uint64_t startTimestamp;
  uint64_t endTimestamp;
  std::vector<double> values;
};

struct FakeCuptiState {
  int major{9};
  int minor{0};
  size_t numPasses{1};
  CUpti_PmSampling_DecodeStopReason decodeStopReason{
      CUPTI_PM_SAMPLING_DECODE_STOP_REASON_END_OF_RECORDS};
  std::vector<FakeSample> decodedSamples;

  std::vector<std::string> calls;

  std::vector<std::string> configMetricNames;
  size_t enabledDeviceIndex{0};
  CUptiResult setConfigResult{CUPTI_SUCCESS};
  size_t hardwareBufferSize{0};
  uint64_t samplingInterval{0};
  CUpti_PmSampling_TriggerMode triggerMode{};
  CUpti_PmSampling_HardwareBuffer_AppendMode appendMode{};
  std::vector<std::string> counterDataMetricNames;
  uint32_t maxSamples{0};

  std::byte hostObject{};
  std::byte samplingObject{};
};

FakeCuptiState& fakeCupti() {
  static FakeCuptiState state;
  return state;
}

CUpti_Profiler_Host_Object* fakeHostObject() {
  return reinterpret_cast<CUpti_Profiler_Host_Object*>(&fakeCupti().hostObject);
}

CUpti_PmSampling_Object* fakeSamplingObject() {
  return reinterpret_cast<CUpti_PmSampling_Object*>(
      &fakeCupti().samplingObject);
}

void recordCall(std::string call) {
  fakeCupti().calls.push_back(std::move(call));
}

void clearCalls() {
  fakeCupti().calls.clear();
}

void expectCalls(std::initializer_list<std::string> expected) {
  EXPECT_EQ(fakeCupti().calls, std::vector<std::string>(expected));
}

std::vector<std::string> copyMetricNames(
    const char* const* metricNames,
    size_t count) {
  std::vector<std::string> result;
  result.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    result.emplace_back(metricNames[i]);
  }
  return result;
}

CuptiPMSamplingConfig makeConfig(
    std::chrono::nanoseconds samplingInterval = 500us,
    int32_t deviceId = 0,
    std::vector<std::string> metricNames = {"sm__cycles_active.avg"}) {
  return CuptiPMSamplingConfig{
      deviceId, std::move(metricNames), samplingInterval};
}

void configureForDevice(
    int major,
    int minor,
    std::chrono::nanoseconds samplingInterval) {
  fakeCupti().major = major;
  fakeCupti().minor = minor;
  CuptiPMSamplingApi api;
  api.configure(makeConfig(samplingInterval));
  api.disable();
}

class CuptiPMSamplingApiTest : public testing::Test {
 protected:
  void SetUp() override {
    fakeCupti() = FakeCuptiState{};
  }
};

} // namespace

// CUDA/CUPTI symbol overrides used by the tests below.
extern "C" {

CUptiResult CUPTIAPI
cuptiGetResultString(CUptiResult result, const char** resultString) {
  static_cast<void>(result);
  *resultString = "mock CUPTI error";
  return CUPTI_SUCCESS;
}

CUptiResult CUPTIAPI
cuptiProfilerInitialize(CUpti_Profiler_Initialize_Params* params) {
  static_cast<void>(params);
  return CUPTI_SUCCESS;
}

CUptiResult CUPTIAPI
cuptiDeviceGetChipName(CUpti_Device_GetChipName_Params* params) {
  params->pChipName = "mock-chip";
  return CUPTI_SUCCESS;
}

cudaError_t CUDARTAPI
cudaGetDeviceProperties(cudaDeviceProp* properties, int device) {
  static_cast<void>(device);
  *properties = cudaDeviceProp{};
  properties->major = fakeCupti().major;
  properties->minor = fakeCupti().minor;
  return cudaSuccess;
}

const char* CUDARTAPI cudaGetErrorString(cudaError_t error) {
  static_cast<void>(error);
  return "mock CUDA error";
}

CUptiResult CUPTIAPI cuptiPmSamplingGetCounterAvailability(
    CUpti_PmSampling_GetCounterAvailability_Params* params) {
  if (params->pCounterAvailabilityImage == nullptr) {
    params->counterAvailabilityImageSize = 1;
  }
  return CUPTI_SUCCESS;
}

CUptiResult CUPTIAPI
cuptiProfilerHostInitialize(CUpti_Profiler_Host_Initialize_Params* params) {
  params->pHostObject = fakeHostObject();
  return CUPTI_SUCCESS;
}

CUptiResult CUPTIAPI cuptiProfilerHostConfigAddMetrics(
    CUpti_Profiler_Host_ConfigAddMetrics_Params* params) {
  fakeCupti().configMetricNames =
      copyMetricNames(params->ppMetricNames, params->numMetrics);
  return CUPTI_SUCCESS;
}

CUptiResult CUPTIAPI cuptiProfilerHostGetConfigImageSize(
    CUpti_Profiler_Host_GetConfigImageSize_Params* params) {
  params->configImageSize = 1;
  return CUPTI_SUCCESS;
}

CUptiResult CUPTIAPI cuptiProfilerHostGetConfigImage(
    CUpti_Profiler_Host_GetConfigImage_Params* params) {
  static_cast<void>(params);
  return CUPTI_SUCCESS;
}

CUptiResult CUPTIAPI cuptiProfilerHostGetNumOfPasses(
    CUpti_Profiler_Host_GetNumOfPasses_Params* params) {
  params->numOfPasses = fakeCupti().numPasses;
  return CUPTI_SUCCESS;
}

CUptiResult CUPTIAPI
cuptiPmSamplingEnable(CUpti_PmSampling_Enable_Params* params) {
  fakeCupti().enabledDeviceIndex = params->deviceIndex;
  params->pPmSamplingObject = fakeSamplingObject();
  return CUPTI_SUCCESS;
}

CUptiResult CUPTIAPI
cuptiPmSamplingSetConfig(CUpti_PmSampling_SetConfig_Params* params) {
  if (fakeCupti().setConfigResult != CUPTI_SUCCESS) {
    return fakeCupti().setConfigResult;
  }
  fakeCupti().hardwareBufferSize = params->hardwareBufferSize;
  fakeCupti().samplingInterval = params->samplingInterval;
  fakeCupti().triggerMode = params->triggerMode;
  fakeCupti().appendMode = params->hwBufferAppendMode;
  return CUPTI_SUCCESS;
}

CUptiResult CUPTIAPI cuptiPmSamplingGetCounterDataSize(
    CUpti_PmSampling_GetCounterDataSize_Params* params) {
  fakeCupti().counterDataMetricNames =
      copyMetricNames(params->pMetricNames, params->numMetrics);
  fakeCupti().maxSamples = params->maxSamples;
  params->counterDataSize = 1;
  return CUPTI_SUCCESS;
}

CUptiResult CUPTIAPI cuptiPmSamplingCounterDataImageInitialize(
    CUpti_PmSampling_CounterDataImage_Initialize_Params* params) {
  static_cast<void>(params);
  recordCall("counterDataImageInitialize");
  return CUPTI_SUCCESS;
}

CUptiResult CUPTIAPI
cuptiPmSamplingStart(CUpti_PmSampling_Start_Params* params) {
  static_cast<void>(params);
  recordCall("samplingStart");
  return CUPTI_SUCCESS;
}

CUptiResult CUPTIAPI
cuptiPmSamplingDecodeData(CUpti_PmSampling_DecodeData_Params* params) {
  params->decodeStopReason = fakeCupti().decodeStopReason;
  params->overflow = 0;
  return CUPTI_SUCCESS;
}

CUptiResult CUPTIAPI cuptiPmSamplingGetCounterDataInfo(
    CUpti_PmSampling_GetCounterDataInfo_Params* params) {
  params->numCompletedSamples = fakeCupti().decodedSamples.size();
  return CUPTI_SUCCESS;
}

CUptiResult CUPTIAPI cuptiPmSamplingCounterDataGetSampleInfo(
    CUpti_PmSampling_CounterData_GetSampleInfo_Params* params) {
  const auto& sample = fakeCupti().decodedSamples[params->sampleIndex];
  params->startTimestamp = sample.startTimestamp;
  params->endTimestamp = sample.endTimestamp;
  return CUPTI_SUCCESS;
}

CUptiResult CUPTIAPI cuptiProfilerHostEvaluateToGpuValues(
    CUpti_Profiler_Host_EvaluateToGpuValues_Params* params) {
  const auto& values = fakeCupti().decodedSamples[params->rangeIndex].values;
  std::copy(values.begin(), values.end(), params->pMetricValues);
  return CUPTI_SUCCESS;
}

CUptiResult CUPTIAPI cuptiPmSamplingStop(CUpti_PmSampling_Stop_Params* params) {
  static_cast<void>(params);
  recordCall("samplingStop");
  return CUPTI_SUCCESS;
}

CUptiResult CUPTIAPI
cuptiPmSamplingDisable(CUpti_PmSampling_Disable_Params* params) {
  static_cast<void>(params);
  recordCall("samplingDisable");
  return CUPTI_SUCCESS;
}

CUptiResult CUPTIAPI
cuptiProfilerHostDeinitialize(CUpti_Profiler_Host_Deinitialize_Params* params) {
  static_cast<void>(params);
  recordCall("hostDeinitialize");
  return CUPTI_SUCCESS;
}

CUptiResult CUPTIAPI
cuptiProfilerDeInitialize(CUpti_Profiler_DeInitialize_Params* params) {
  static_cast<void>(params);
  recordCall("profilerDeInitialize");
  return CUPTI_SUCCESS;
}

} // extern "C"

TEST_F(CuptiPMSamplingApiTest, RequiresConfigurationForSamplingOperations) {
  CuptiPMSamplingApi api;
  std::vector<CuptiPMSample> samples;

  EXPECT_THROW(api.start(), std::runtime_error);
  EXPECT_THROW(api.decode(samples), std::runtime_error);
  EXPECT_THROW(api.stop(), std::runtime_error);
  EXPECT_NO_THROW(api.disable());
  EXPECT_TRUE(samples.empty());
  EXPECT_TRUE(fakeCupti().calls.empty());
}

TEST_F(CuptiPMSamplingApiTest, ConfiguresDeviceMetricsAndBuffers) {
  const auto config = makeConfig(
      250us,
      /*deviceId=*/2,
      {"sm__cycles_active.avg", "dram__bytes_read.sum"});
  CuptiPMSamplingApi api;

  api.configure(config);

  EXPECT_EQ(fakeCupti().configMetricNames, config.metricNames);
  EXPECT_EQ(fakeCupti().enabledDeviceIndex, 2);
  EXPECT_EQ(fakeCupti().hardwareBufferSize, 64 * 1024 * 1024);
  EXPECT_EQ(fakeCupti().samplingInterval, 250'000);
  EXPECT_EQ(
      fakeCupti().triggerMode,
      CUPTI_PM_SAMPLING_TRIGGER_MODE_GPU_TIME_INTERVAL);
  EXPECT_EQ(
      fakeCupti().appendMode,
      CUPTI_PM_SAMPLING_HARDWARE_BUFFER_APPEND_MODE_KEEP_LATEST);
  EXPECT_EQ(fakeCupti().counterDataMetricNames, config.metricNames);
  EXPECT_EQ(fakeCupti().maxSamples, 1024);
  api.disable();
}

TEST_F(CuptiPMSamplingApiTest, UsesFixedSysclkIntervalOnGa100) {
  configureForDevice(8, 0, 0ns);

  EXPECT_EQ(
      fakeCupti().triggerMode,
      CUPTI_PM_SAMPLING_TRIGGER_MODE_GPU_SYSCLK_INTERVAL);
  EXPECT_EQ(fakeCupti().samplingInterval, 1'000'000);
}

TEST_F(CuptiPMSamplingApiTest, UsesRequestedTimeIntervalOnGa10xAndNewer) {
  for (const auto& [major, minor] : {std::pair{8, 6}, std::pair{9, 0}}) {
    SCOPED_TRACE(testing::Message() << major << "." << minor);
    configureForDevice(major, minor, 500us);

    EXPECT_EQ(
        fakeCupti().triggerMode,
        CUPTI_PM_SAMPLING_TRIGGER_MODE_GPU_TIME_INTERVAL);
    EXPECT_EQ(fakeCupti().samplingInterval, 500'000);
  }
}

TEST_F(CuptiPMSamplingApiTest, RejectsUnsupportedComputeCapabilities) {
  for (const auto& [major, minor] : {std::pair{7, 5}, std::pair{8, 5}}) {
    SCOPED_TRACE(testing::Message() << major << "." << minor);
    EXPECT_THROW(configureForDevice(major, minor, 500us), std::runtime_error);
  }
}

TEST_F(CuptiPMSamplingApiTest, RejectsNonpositiveTimeIntervals) {
  EXPECT_THROW(configureForDevice(8, 6, 0ns), std::runtime_error);
}

TEST_F(CuptiPMSamplingApiTest, RejectsMultipassConfigurationBeforeEnabling) {
  fakeCupti().numPasses = 2;
  CuptiPMSamplingApi api;

  EXPECT_THROW(api.configure(makeConfig()), std::runtime_error);

  clearCalls();
  api.disable();
  expectCalls({"hostDeinitialize", "profilerDeInitialize"});
}

TEST_F(CuptiPMSamplingApiTest, CleansUpAfterCuptiConfigurationFailure) {
  fakeCupti().setConfigResult = CUPTI_ERROR_UNKNOWN;
  CuptiPMSamplingApi api;

  EXPECT_THROW(api.configure(makeConfig()), std::runtime_error);

  clearCalls();
  api.disable();
  expectCalls({"samplingDisable", "hostDeinitialize", "profilerDeInitialize"});
}

TEST_F(CuptiPMSamplingApiTest, RejectsReconfigurationUntilDisabled) {
  CuptiPMSamplingApi api;
  api.configure(makeConfig());

  EXPECT_THROW(
      api.configure(makeConfig(750us, 1, {"dram__bytes_read.sum"})),
      std::runtime_error);

  api.disable();
  api.configure(makeConfig(750us, 1, {"dram__bytes_read.sum"}));
  EXPECT_EQ(
      fakeCupti().configMetricNames,
      (std::vector<std::string>{"dram__bytes_read.sum"}));
  api.disable();
}

TEST_F(CuptiPMSamplingApiTest, StartsAndStopsConfiguredSampler) {
  CuptiPMSamplingApi api;
  api.configure(makeConfig());
  clearCalls();

  api.start();
  api.stop();

  expectCalls({"samplingStart", "samplingStop"});
  api.disable();
}

TEST_F(CuptiPMSamplingApiTest, DecodesCompletedSamplesAndAppendsThem) {
  const auto config = makeConfig(
      500us,
      /*deviceId=*/0,
      {"sm__cycles_active.avg", "dram__bytes_read.sum"});
  fakeCupti().decodedSamples = {
      FakeSample{100, 120, {1.25, 2048.0}},
      FakeSample{130, 170, {2.5, 4096.0}},
  };
  CuptiPMSamplingApi api;
  api.configure(config);
  std::vector<CuptiPMSample> samples{
      CuptiPMSample{1, 2, std::vector<double>{3.0}}};
  clearCalls();

  EXPECT_TRUE(api.decode(samples));

  expectCalls({"counterDataImageInitialize"});
  ASSERT_EQ(samples.size(), 3);
  EXPECT_EQ(samples[0].rawStartTimestamp, 1);
  EXPECT_EQ(samples[0].rawEndTimestamp, 2);
  EXPECT_EQ(samples[0].values, (std::vector<double>{3.0}));
  EXPECT_EQ(samples[1].rawStartTimestamp, 100);
  EXPECT_EQ(samples[1].rawEndTimestamp, 120);
  EXPECT_EQ(samples[1].values, (std::vector<double>{1.25, 2048.0}));
  EXPECT_EQ(samples[2].rawStartTimestamp, 130);
  EXPECT_EQ(samples[2].rawEndTimestamp, 170);
  EXPECT_EQ(samples[2].values, (std::vector<double>{2.5, 4096.0}));
  api.disable();
}

TEST_F(CuptiPMSamplingApiTest, ReportsFullCounterDataAsUndrained) {
  fakeCupti().decodedSamples = {
      FakeSample{100, 120, {1.25}},
  };
  CuptiPMSamplingApi api;
  api.configure(makeConfig());
  std::vector<CuptiPMSample> samples;

  fakeCupti().decodeStopReason =
      CUPTI_PM_SAMPLING_DECODE_STOP_REASON_COUNTER_DATA_FULL;
  EXPECT_FALSE(api.decode(samples));
  ASSERT_EQ(samples.size(), 1);
  EXPECT_EQ(samples[0].rawStartTimestamp, 100);
  EXPECT_EQ(samples[0].rawEndTimestamp, 120);
  EXPECT_EQ(samples[0].values, (std::vector<double>{1.25}));
  api.disable();
}

TEST_F(CuptiPMSamplingApiTest, DisableIsOrderedAndIdempotent) {
  CuptiPMSamplingApi api;
  api.configure(makeConfig());
  clearCalls();

  api.disable();

  expectCalls({"samplingDisable", "hostDeinitialize", "profilerDeInitialize"});
  clearCalls();
  api.disable();
  EXPECT_TRUE(fakeCupti().calls.empty());
  EXPECT_THROW(api.start(), std::runtime_error);
}

TEST_F(CuptiPMSamplingApiTest, DestructorDisablesConfiguredSampler) {
  {
    CuptiPMSamplingApi api;
    api.configure(makeConfig());
    clearCalls();
  }

  expectCalls({"samplingDisable", "hostDeinitialize", "profilerDeInitialize"});
}
