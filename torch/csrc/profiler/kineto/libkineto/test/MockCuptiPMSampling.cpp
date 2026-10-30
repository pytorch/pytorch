/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "test/MockCuptiPMSampling.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include <cupti_profiler_host.h>
#include <cupti_profiler_target.h>
#include <cupti_target.h>

namespace libkineto::test {

MockCuptiState& mockCupti() {
  static MockCuptiState state;
  return state;
}

} // namespace libkineto::test

using libkineto::test::mockCupti;

namespace {

CUpti_Profiler_Host_Object* mockHostObject() {
  return reinterpret_cast<CUpti_Profiler_Host_Object*>(&mockCupti().hostObject);
}

CUpti_PmSampling_Object* mockSamplingObject() {
  return reinterpret_cast<CUpti_PmSampling_Object*>(
      &mockCupti().samplingObject);
}

void recordCall(std::string call) {
  mockCupti().calls.push_back(std::move(call));
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

} // namespace

// CUDA/CUPTI symbol overrides shared by the API and integration tests.
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
  properties->major = mockCupti().major;
  properties->minor = mockCupti().minor;
  return cudaSuccess;
}

cudaError_t CUDARTAPI
cudaDeviceGetAttribute(int* value, cudaDeviceAttr attribute, int device) {
  EXPECT_EQ(attribute, cudaDevAttrClockRate);
  mockCupti().clockRateDevice = device;
  *value = mockCupti().clockRate;
  return mockCupti().clockRateResult;
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
  params->pHostObject = mockHostObject();
  return CUPTI_SUCCESS;
}

CUptiResult CUPTIAPI cuptiProfilerHostConfigAddMetrics(
    CUpti_Profiler_Host_ConfigAddMetrics_Params* params) {
  mockCupti().configMetricNames =
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
  params->numOfPasses = mockCupti().numPasses;
  return CUPTI_SUCCESS;
}

CUptiResult CUPTIAPI
cuptiPmSamplingEnable(CUpti_PmSampling_Enable_Params* params) {
  mockCupti().enabledDeviceIndex = params->deviceIndex;
  params->pPmSamplingObject = mockSamplingObject();
  return CUPTI_SUCCESS;
}

CUptiResult CUPTIAPI
cuptiPmSamplingSetConfig(CUpti_PmSampling_SetConfig_Params* params) {
  if (mockCupti().setConfigResult != CUPTI_SUCCESS) {
    return mockCupti().setConfigResult;
  }
  mockCupti().hardwareBufferSize = params->hardwareBufferSize;
  mockCupti().samplingInterval = params->samplingInterval;
  mockCupti().triggerMode = params->triggerMode;
  mockCupti().appendMode = params->hwBufferAppendMode;
  return CUPTI_SUCCESS;
}

CUptiResult CUPTIAPI cuptiPmSamplingGetCounterDataSize(
    CUpti_PmSampling_GetCounterDataSize_Params* params) {
  mockCupti().counterDataMetricNames =
      copyMetricNames(params->pMetricNames, params->numMetrics);
  mockCupti().maxSamples = params->maxSamples;
  params->counterDataSize = mockCupti().counterDataSize;
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
  params->decodeStopReason = mockCupti().decodeStopReason;
  params->overflow = 0;
  return CUPTI_SUCCESS;
}

CUptiResult CUPTIAPI cuptiPmSamplingGetCounterDataInfo(
    CUpti_PmSampling_GetCounterDataInfo_Params* params) {
  params->numCompletedSamples = mockCupti().decodedSamples.size();
  return CUPTI_SUCCESS;
}

CUptiResult CUPTIAPI cuptiPmSamplingCounterDataGetSampleInfo(
    CUpti_PmSampling_CounterData_GetSampleInfo_Params* params) {
  const auto& sample = mockCupti().decodedSamples[params->sampleIndex];
  params->startTimestamp = sample.startTimestamp;
  params->endTimestamp = sample.endTimestamp;
  return CUPTI_SUCCESS;
}

CUptiResult CUPTIAPI cuptiProfilerHostEvaluateToGpuValues(
    CUpti_Profiler_Host_EvaluateToGpuValues_Params* params) {
  const auto& values = mockCupti().decodedSamples[params->rangeIndex].values;
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
